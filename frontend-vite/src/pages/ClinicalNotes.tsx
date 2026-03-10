// gateway/src/pages/ClinicalNotes.tsx — Full Redesign
// 3-panel workflow: Setup | Audio + Transcript | SOAP Editor
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
  IconChevronDown,
  IconChevronUp,
  IconUser,
  IconPlus,
  IconStethoscope,
  IconEdit,
  IconLayoutGrid,
  IconCode,
} from '@tabler/icons-react';
import api from '../utils/api';
import { useAuthStore } from '@store/authStore';

// ─── Types ────────────────────────────────────────────────────────────────────

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
  error?: string;
  reviewStartTime?: number;
  dialect?: string;
}

interface SoapSections {
  subjective: string;
  objective: string;
  assessment: string;
  plan: string;
}

interface MetricsDashboard {
  overview: {
    totalNotes: number;
    acceptanceRate: number;
    avgEditDistance: number;
    avgReviewTime: number;
  };
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

const getSupportedMimeType = (): string => {
  const candidates = [
    'audio/webm;codecs=opus',
    'audio/webm',
    'audio/ogg;codecs=opus',
    'audio/mp4',
  ];
  for (const type of candidates) {
    if (typeof MediaRecorder !== 'undefined' && MediaRecorder.isTypeSupported(type))
      return type;
  }
  return '';
};

const fileToBase64 = (file: File): Promise<string> =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () =>
      resolve((String(reader.result || '')).split(',')[1] || '');
    reader.onerror = () => reject(new Error('Failed to read file'));
    reader.readAsDataURL(file);
  });

const formatTime = (s: number) =>
  `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;

const mapDialect = (d: string) => {
  if (d === 'egyptian') return 'egypt';
  if (d === 'levantine') return 'levant';
  if (d === 'gulf') return 'gulf';
  if (d === 'msa') return 'msa';
  return undefined;
};

const buildSoapText = (s: { subjective?: any; objective?: any; assessment?: any; plan?: any }) =>
  [
    `Subjective:\n${s.subjective || ''}`,
    `Objective:\n${s.objective || ''}`,
    `Assessment:\n${s.assessment || ''}`,
    `Plan:\n${s.plan || ''}`,
  ].join('\n\n');

const parseSoapSections = (soapJson: any, soapNote: string): SoapSections => {
  if (soapJson) {
    const src = soapJson.soap_note || soapJson;
    // Only use the JSON path when keys are plain strings (not nested objects from soap_json template)
    const s = typeof src.subjective === 'string' ? src.subjective : typeof src.S === 'string' ? src.S : '';
    const o = typeof src.objective === 'string' ? src.objective : typeof src.O === 'string' ? src.O : '';
    const a = typeof src.assessment === 'string' ? src.assessment : typeof src.A === 'string' ? src.A : '';
    const p = typeof src.plan === 'string' ? src.plan : typeof src.P === 'string' ? src.P : '';
    if (s || o || a || p) return { subjective: s, objective: o, assessment: a, plan: p };
  }
  // Fallback: parse from the formatted soapNote text built from the response's top-level string fields
  const text = soapNote || '';
  const getSection = (label: string, next: string) => {
    const start = text.indexOf(`${label}:\n`);
    if (start === -1) return '';
    const body = text.slice(start + label.length + 2);
    const end = next ? body.indexOf(`${next}:\n`) : body.length;
    return (end === -1 ? body : body.slice(0, end)).trim();
  };
  return {
    subjective: getSection('Subjective', 'Objective'),
    objective: getSection('Objective', 'Assessment'),
    assessment: getSection('Assessment', 'Plan'),
    plan: getSection('Plan', ''),
  };
};

const sanitizeError = (err: any): string => {
  const msg = String(err?.message || '').toLowerCase();
  if (msg.includes('downstream') || msg.includes('502') || msg.includes('503') || msg.includes('econnrefused'))
    return 'تعذر الاتصال بالخادم. تأكد من تشغيل الخدمة وحاول مجدداً.';
  if (msg.includes('401') || msg.includes('unauthorized')) return 'انتهت صلاحية الجلسة، يرجى تسجيل الدخول مجدداً';
  if (msg.includes('409') || msg.includes('conflict')) return 'يوجد سجل بنفس البيانات';
  if (msg.includes('400') || msg.includes('bad request')) return 'بيانات غير صحيحة، تحقق من المدخلات';
  return err?.message || 'حدث خطأ غير متوقع';
};

// ─── SOAP Section Config ──────────────────────────────────────────────────────

const SOAP_SECTIONS = [
  {
    key: 'subjective' as const,
    label: 'S — ذاتي',
    desc: 'شكوى المريض وأعراضه',
    border: 'border-blue-500/40',
    bg: 'bg-blue-500/[0.07]',
    header: 'bg-blue-500/20',
    ring: 'focus:ring-blue-500',
    dot: 'bg-blue-400',
    badge: 'text-blue-300',
    placeholder: 'شكوى المريض الرئيسية، بداية الأعراض ومدتها...',
  },
  {
    key: 'objective' as const,
    label: 'O — موضوعي',
    desc: 'الفحص والعلامات الحيوية',
    border: 'border-emerald-500/40',
    bg: 'bg-emerald-500/[0.07]',
    header: 'bg-emerald-500/20',
    ring: 'focus:ring-emerald-500',
    dot: 'bg-emerald-400',
    badge: 'text-emerald-300',
    placeholder: 'الفحص السريري، العلامات الحيوية، نتائج الفحوصات...',
  },
  {
    key: 'assessment' as const,
    label: 'A — تقييم',
    desc: 'التشخيص والتفسير',
    border: 'border-amber-500/40',
    bg: 'bg-amber-500/[0.07]',
    header: 'bg-amber-500/20',
    ring: 'focus:ring-amber-500',
    dot: 'bg-amber-400',
    badge: 'text-amber-300',
    placeholder: 'التشخيص الرئيسي، التشخيص التفريقي...',
  },
  {
    key: 'plan' as const,
    label: 'P — خطة',
    desc: 'العلاج والمتابعة',
    border: 'border-rose-500/40',
    bg: 'bg-rose-500/[0.07]',
    header: 'bg-rose-500/20',
    ring: 'focus:ring-rose-500',
    dot: 'bg-rose-400',
    badge: 'text-rose-300',
    placeholder: 'الأدوية الموصوفة، الإحالات، موعد المتابعة...',
  },
] as const;

// ─── Built-in Template Library ─────────────────────────────────────────────

const BUILTIN_TEMPLATES = [
  { id: 'standard-soap',  name: 'SOAP القياسي',      nameEn: 'Standard SOAP',  icon: '📋', color: 'purple',  desc: 'القالب العام: ذاتي • موضوعي • تقييم • خطة',     template: { style: 'standard',    sections: ['subjective','objective','assessment','plan'] } },
  { id: 'compact-soap',   name: 'SOAP المختصر',       nameEn: 'Compact SOAP',   icon: '⚡', color: 'blue',    desc: 'نسخة موجزة للزيارات السريعة',                  template: { style: 'compact',     sections: ['subjective','objective','assessment','plan'] } },
  { id: 'psychiatric',    name: 'الطب النفسي',         nameEn: 'Psychiatric',    icon: '🧠', color: 'indigo',  desc: 'يشمل تقييم الحالة العقلية والمخاطر',            template: { style: 'psychiatric', sections: ['subjective','objective','assessment','plan'], extras: ['mentalStatus','risk'] } },
  { id: 'pediatric',      name: 'طب الأطفال',          nameEn: 'Pediatric',      icon: '👶', color: 'emerald', desc: 'مخصص للمرضى دون سن 18',                         template: { style: 'pediatric',   sections: ['subjective','objective','assessment','plan'] } },
  { id: 'emergency',      name: 'الطوارئ',             nameEn: 'Emergency',      icon: '🚨', color: 'red',     desc: 'قالب سريع لحالات الإسعاف والطوارئ',             template: { style: 'emergency',   sections: ['subjective','objective','assessment','plan'] } },
  { id: 'dap',            name: 'قالب DAP',            nameEn: 'DAP Note',       icon: '📄', color: 'amber',   desc: 'بيانات • تقييم • خطة (شائع في الإرشاد)',        template: { style: 'dap',         sections: ['data','assessment','plan'] } },
  { id: 'birp',           name: 'قالب BIRP',           nameEn: 'BIRP Note',      icon: '🔍', color: 'teal',    desc: 'سلوك • تدخل • استجابة • خطة',                  template: { style: 'birp',        sections: ['behavior','intervention','response','plan'] } },
  { id: 'progress',       name: 'ملاحظة المتابعة',      nameEn: 'Progress Note',  icon: '📈', color: 'cyan',    desc: 'متابعة تطور حالة المريض وعلاجه',                template: { style: 'progress',    sections: ['subjective','objective','assessment','plan'] } },
  { id: 'cardiology',     name: 'القلب والأوعية',       nameEn: 'Cardiology',     icon: '❤️', color: 'rose',    desc: 'يشمل مؤشرات القلب التفصيلية',                  template: { style: 'cardiology',  sections: ['subjective','objective','assessment','plan'] } },
  { id: 'dermatology',    name: 'الجلدية',              nameEn: 'Dermatology',    icon: '🔬', color: 'orange',  desc: 'وصف تفصيلي للحالة الجلدية',                    template: { style: 'dermatology', sections: ['subjective','objective','assessment','plan'] } },
  { id: 'obstetrics',     name: 'النساء والتوليد',       nameEn: 'OB/GYN',         icon: '🤰', color: 'pink',    desc: 'مخصص لطب التوليد وأمراض النساء',               template: { style: 'obstetrics',  sections: ['subjective','objective','assessment','plan'] } },
  { id: 'orthopedics',    name: 'العظام والمفاصل',       nameEn: 'Orthopedics',    icon: '🦴', color: 'slate',   desc: 'يشمل تقييم الحركة والألم',                      template: { style: 'orthopedics', sections: ['subjective','objective','assessment','plan'] } },
] as const;

// ─── Main Component ───────────────────────────────────────────────────────────

export default function ClinicalNotes() {
  const { userId, token } = useAuthStore();

  // Visit setup
  const [patients, setPatients] = useState<Array<{ id: string; displayName?: string; externalId?: string }>>([]);
  const [patientId, setPatientId] = useState('');
  const [patientName, setPatientName] = useState('');
  const [practitionerId, setPractitionerId] = useState('');
  const [providerName, setProviderName] = useState('');
  const [dateOfVisit, setDateOfVisit] = useState(() => new Date().toISOString().split('T')[0]);
  const [templates, setTemplates] = useState<Array<{ id: string; name: string }>>([]);
  const [selectedTemplateId, setSelectedTemplateId] = useState('');
  const [selectedDialect, setSelectedDialect] = useState('auto');

  // Add patient
  const [showAddPatient, setShowAddPatient] = useState(false);
  const [newPatientName, setNewPatientName] = useState('');
  const [newPatientExternalId, setNewPatientExternalId] = useState('');
  const [patientStatus, setPatientStatus] = useState<'idle' | 'loading' | 'error'>('idle');
  const [patientError, setPatientError] = useState('');

  // Recordings
  const [recordings, setRecordings] = useState<AudioRecording[]>([]);
  const [selectedRecording, setSelectedRecording] = useState<AudioRecording | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);

  // SOAP editing
  const [soapSections, setSoapSections] = useState<SoapSections>({
    subjective: '', objective: '', assessment: '', plan: '',
  });
  const [sectionRecording, setSectionRecording] = useState<keyof SoapSections | null>(null);
  const [sectionRecordingTime, setSectionRecordingTime] = useState(0);
  const [sectionUpdating, setSectionUpdating] = useState<keyof SoapSections | null>(null);

  // Template manager
  const [showTemplateManager, setShowTemplateManager] = useState(false);
  const [templateTab, setTemplateTab] = useState<'builtin' | 'upload' | 'custom'>('builtin');
  const [customTemplateName, setCustomTemplateName] = useState('');
  const [customTemplateText, setCustomTemplateText] = useState('');
  const [templateSaving, setTemplateSaving] = useState(false);
  const [templateError, setTemplateError] = useState('');
  const [selectedTemplateJson, setSelectedTemplateJson] = useState<Record<string, any> | null>(null);
  const [selectedTemplateName, setSelectedTemplateName] = useState('');

  // Patient files
  const [showPatientFiles, setShowPatientFiles] = useState(false);
  const [patientDocs, setPatientDocs] = useState<any[]>([]);

  // Metrics
  const [showMetrics, setShowMetrics] = useState(false);
  const [metrics, setMetrics] = useState<MetricsDashboard | null>(null);
  const [metricsLoading, setMetricsLoading] = useState(false);

  // UI
  const [isSaving, setIsSaving] = useState(false);
  const [toast, setToast] = useState<{ type: 'success' | 'error'; message: string } | null>(null);

  // Refs
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const sectionRecorderRef = useRef<MediaRecorder | null>(null);
  const sectionAudioChunksRef = useRef<Blob[]>([]);
  const sectionTimerRef = useRef<number | null>(null);
  const recordingDurationRef = useRef(0);
  const templateFileInputRef = useRef<HTMLInputElement>(null);

  // ─── Effects ─────────────────────────────────────────────────────────────

  useEffect(() => {
    if (userId) setPractitionerId(userId);
  }, [userId]);

  useEffect(() => {
    if (token) {
      loadTemplates();
      loadPatients();
    }
  }, [token]);

  useEffect(() => {
    if (selectedRecording?.status === 'completed') {
      setSoapSections(parseSoapSections(
        selectedRecording.soapJson,
        selectedRecording.soapNote || '',
      ));
    } else {
      setSoapSections({ subjective: '', objective: '', assessment: '', plan: '' });
    }
  }, [selectedRecording?.id, selectedRecording?.status]);

  useEffect(() => {
    if (patientId) loadPatientDocuments(patientId);
    else setPatientDocs([]);
  }, [patientId]);

  // ─── Toast ────────────────────────────────────────────────────────────────

  const showToast = (type: 'success' | 'error', message: string) => {
    setToast({ type, message });
    setTimeout(() => setToast(null), 4000);
  };

  // ─── Template save helpers ──────────────────────────────────────────────

  const createCustomTemplate = async (name: string, templateObj: Record<string, any>) => {
    setTemplateSaving(true);
    setTemplateError('');
    try {
      await api.createSoapTemplate({ name, template: templateObj });
      await loadTemplates();
      setShowTemplateManager(false);
      setCustomTemplateName('');
      setCustomTemplateText('');
      showToast('success', `تم حفظ القالب "${name}" بنجاح`);
    } catch (err: any) {
      setTemplateError(sanitizeError(err));
    } finally {
      setTemplateSaving(false);
    }
  };

  const handleTemplateFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const text = await file.text();
      const json = JSON.parse(text);
      const name = customTemplateName.trim() || file.name.replace(/\.[^.]+$/, '');
      await createCustomTemplate(name, json);
    } catch {
      setTemplateError('الملف غير صالح. يجب أن يكون بتنسيق JSON صحيح.');
    }
    if (templateFileInputRef.current) templateFileInputRef.current.value = '';
  };

  // ─── Load data ────────────────────────────────────────────────────────────

  const loadTemplates = async () => {
    try {
      const data = await api.listSoapTemplates();
      const list = data.templates || [];
      setTemplates(list);
      if (list.length > 0 && !selectedTemplateId) setSelectedTemplateId(list[0].id);
    } catch { /* silent */ }
  };

  const loadPatients = async () => {
    try {
      setPatientStatus('loading');
      const data = await api.listPatients();
      const list = data.patients || [];
      setPatients(list);
      if (list.length > 0) {
        setPatientId(list[0].id);
        if (list[0].displayName) setPatientName(list[0].displayName);
      }
      setPatientStatus('idle');
      setPatientError('');
    } catch {
      setPatientStatus('error');
      setPatientError('تعذر تحميل قائمة المرضى');
    }
  };

  const loadPatientDocuments = async (id: string) => {
    try {
      const data = await api.listPatientDocuments(id);
      setPatientDocs(data.documents || []);
    } catch { setPatientDocs([]); }
  };

  const loadMetrics = async () => {
    try {
      setMetricsLoading(true);
      const data = await api.getClinicalMetricsDashboard();
      setMetrics(data);
    } catch { /* silent */ } finally {
      setMetricsLoading(false);
    }
  };

  // ─── Patient ──────────────────────────────────────────────────────────────

  const createPatient = async () => {
    if (!newPatientName.trim()) { setPatientError('أدخل اسم المريض'); return; }
    try {
      setPatientStatus('loading');
      const created = await api.createPatient({
        displayName: newPatientName.trim(),
        externalId: newPatientExternalId.trim() || undefined,
      });
      setPatients((p) => [{ id: created.id, displayName: created.displayName, externalId: created.externalId }, ...p]);
      setPatientId(created.id);
      setPatientName(created.displayName || newPatientName.trim());
      setNewPatientName('');
      setNewPatientExternalId('');
      setShowAddPatient(false);
      setPatientStatus('idle');
      setPatientError('');
    } catch (err: any) {
      setPatientStatus('error');
      setPatientError(sanitizeError(err));
    }
  };

  // ─── Recording ────────────────────────────────────────────────────────────

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mime = getSupportedMimeType();
      const mr = new MediaRecorder(stream, mime ? { mimeType: mime } : {});
      mediaRecorderRef.current = mr;
      audioChunksRef.current = [];
      mr.ondataavailable = (e) => { if (e.data.size > 0) audioChunksRef.current.push(e.data); };
      mr.onstop = async () => {
        const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        const file = new File([blob], `recording-${Date.now()}.webm`, { type: 'audio/webm' });
        const rec: AudioRecording = {
          id: `rec-${Date.now()}`, file,
          duration: recordingDurationRef.current,
          timestamp: new Date(), status: 'pending',
        };
        setRecordings((p) => [rec, ...p]);
        setRecordingTime(0);
        recordingDurationRef.current = 0;
        await processRecording(rec);
      };
      mr.start();
      setIsRecording(true);
      timerRef.current = window.setInterval(() => {
        setRecordingTime((p) => { recordingDurationRef.current = p + 1; return p + 1; });
      }, 1000);
    } catch (err: any) {
      showToast('error', `خطأ في التسجيل: ${err.message}`);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach((t) => t.stop());
      setIsRecording(false);
      if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files?.length) return;
    for (const file of Array.from(files)) {
      const rec: AudioRecording = {
        id: `upload-${Date.now()}-${Math.random()}`, file,
        duration: 0, timestamp: new Date(), status: 'pending',
      };
      setRecordings((p) => [rec, ...p]);
      await processRecording(rec);
    }
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  // ─── Process ──────────────────────────────────────────────────────────────

  const processRecording = async (recording: AudioRecording) => {
    try {
      if (!patientId || !practitionerId) throw new Error('اختر المريض وتأكد من إدخال معرف الطبيب');

      const setStatus = (patch: Partial<AudioRecording>) => {
        setRecordings((p) => p.map((r) => r.id === recording.id ? { ...r, ...patch } : r));
        setSelectedRecording((p) => p?.id === recording.id ? { ...(p ?? recording), ...patch } as AudioRecording : p);
      };

      setStatus({ status: 'processing' });
      setSelectedRecording({ ...recording, status: 'processing' });

      const audioBase64 = await fileToBase64(recording.file);
      const transcriptRes: any = await api.transcribeAudio(
        audioBase64, recording.id, mapDialect(selectedDialect), 'ar', true, false,
      );
      const transcript = transcriptRes.text || '';

      const soapRes = await api.createSOAPNote({
        transcript,
        sessionId: recording.id,
        patientId,
        practitionerId,
        templateId: selectedTemplateId || undefined,
        templateJson: selectedTemplateJson || undefined,
        patientName,
        providerName,
        dateOfVisit,
      });

      const patch: Partial<AudioRecording> = {
        status: 'completed',
        transcript,
        soapNote: buildSoapText(soapRes),
        soapJson: (soapRes as any).soapJson ?? (soapRes as any).soap_json,
        noteId: soapRes.id,
        dialect: selectedDialect,
      };
      setStatus(patch);
      setSelectedRecording({ ...recording, ...patch } as AudioRecording);
    } catch (err: any) {
      const patch = { status: 'error' as const, error: sanitizeError(err) };
      setRecordings((p) => p.map((r) => r.id === recording.id ? { ...r, ...patch } : r));
      if (selectedRecording?.id === recording.id || !selectedRecording) {
        setSelectedRecording({ ...recording, ...patch });
      }
    }
  };

  // ─── Section voice update ─────────────────────────────────────────────────

  const startSectionRecording = async (section: keyof SoapSections) => {
    if (!selectedRecording?.noteId) { showToast('error', 'لا توجد ملاحظة نشطة'); return; }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mime = getSupportedMimeType();
      const mr = new MediaRecorder(stream, mime ? { mimeType: mime } : {});
      sectionRecorderRef.current = mr;
      sectionAudioChunksRef.current = [];
      mr.ondataavailable = (e) => { if (e.data.size > 0) sectionAudioChunksRef.current.push(e.data); };
      mr.onstop = async () => {
        const blob = new Blob(sectionAudioChunksRef.current, { type: 'audio/webm' });
        const file = new File([blob], `section-${Date.now()}.webm`, { type: 'audio/webm' });
        setSectionRecordingTime(0);
        await updateSectionByVoice(section, file);
      };
      mr.start();
      setSectionRecording(section);
      setSectionRecordingTime(0);
      sectionTimerRef.current = window.setInterval(
        () => setSectionRecordingTime((p) => p + 1), 1000,
      );
    } catch (err: any) {
      showToast('error', `خطأ في التسجيل: ${err.message}`);
    }
  };

  const stopSectionRecording = () => {
    if (sectionRecorderRef.current && sectionRecording) {
      sectionRecorderRef.current.stop();
      sectionRecorderRef.current.stream.getTracks().forEach((t) => t.stop());
      setSectionRecording(null);
      if (sectionTimerRef.current) {
        clearInterval(sectionTimerRef.current);
        sectionTimerRef.current = null;
      }
    }
  };

  const updateSectionByVoice = async (section: keyof SoapSections, audioFile: File) => {
    if (!selectedRecording?.noteId) return;
    try {
      setSectionUpdating(section);
      const audioBase64 = await fileToBase64(audioFile);
      const updated = await api.updateSOAPNoteField(selectedRecording.noteId, {
        fieldPath: section,
        audio: audioBase64,
        mode: 'replace',
        valueType: 'string',
        dialect: mapDialect(selectedDialect),
        language: 'ar',
      });
      const newJson = updated.soapJson ?? updated.soap_json ?? selectedRecording.soapJson;
      const newSections = parseSoapSections(newJson, '');
      setSoapSections(newSections);
      const fullUpdated = {
        ...selectedRecording,
        soapJson: newJson,
        soapNote: buildSoapText(newSections),
      };
      setRecordings((p) => p.map((r) => r.id === selectedRecording.id ? fullUpdated : r));
      setSelectedRecording(fullUpdated);
      showToast('success', 'تم تحديث القسم');
    } catch (err: any) {
      showToast('error', `فشل التحديث: ${err.message}`);
    } finally {
      setSectionUpdating(null);
    }
  };

  // ─── Save / Approve ───────────────────────────────────────────────────────

  const saveToEHR = async () => {
    if (!selectedRecording?.noteId) return;
    setIsSaving(true);
    try {
      await api.updateSOAPNoteSections(selectedRecording.noteId, {
        subjective: soapSections.subjective,
        objective: soapSections.objective,
        assessment: soapSections.assessment,
        plan: soapSections.plan,
      });
      await api.approveSOAPNote(selectedRecording.noteId);
      const timeToReview = selectedRecording.reviewStartTime
        ? Math.round((Date.now() - selectedRecording.reviewStartTime) / 1000)
        : 0;
      await api.recordClinicalReview({
        recordingId: selectedRecording.id,
        accepted: true,
        editDistance: 0,
        timeToReview,
        clinicianId: practitionerId || 'unknown',
      }).catch(() => {});
      showToast('success', 'تم الحفظ وإرسال السجل الصحي ✓');
    } catch (err: any) {
      showToast('error', `خطأ في الحفظ: ${err.message}`);
    } finally {
      setIsSaving(false);
    }
  };

  const rejectNote = async () => {
    if (!selectedRecording?.noteId) return;
    try {
      await api.rejectSOAPNote(selectedRecording.noteId);
      await api.recordClinicalReview({
        recordingId: selectedRecording.id,
        accepted: false,
        editDistance: 0,
        timeToReview: 0,
        clinicianId: practitionerId || 'unknown',
      }).catch(() => {});
      showToast('success', 'تم رفض الملاحظة');
    } catch (err: any) {
      showToast('error', err.message);
    }
  };

  const downloadNote = () => {
    const text = buildSoapText(soapSections);
    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `soap-${selectedRecording?.id || 'note'}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // ─── Derived ──────────────────────────────────────────────────────────────

  const hasNote = selectedRecording?.status === 'completed';
  const canSave = hasNote && !!selectedRecording?.noteId;

  // ─── JSX ─────────────────────────────────────────────────────────────────

  return (
    <div
      className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white font-['Tajawal',sans-serif]"
      dir="rtl"
    >
      {/* ── Sticky Header ── */}
      <div className="sticky top-0 z-30 border-b border-white/10 bg-slate-900/80 backdrop-blur-md">
        <div className="max-w-[1600px] mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-purple-500/20 flex items-center justify-center">
              <IconStethoscope className="w-5 h-5 text-purple-400" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-white leading-tight">توثيق الملاحظات السريرية</h1>
              <p className="text-xs text-slate-500">تحويل التسجيلات الصوتية إلى ملاحظات SOAP</p>
            </div>
          </div>
          <button
            onClick={() => { setShowMetrics((p) => !p); if (!metrics) loadMetrics(); }}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium border transition-all ${
              showMetrics
                ? 'bg-purple-500/20 border-purple-500/40 text-purple-300'
                : 'bg-white/5 border-white/10 text-slate-400 hover:bg-white/10 hover:text-white'
            }`}
          >
            <IconChartBar className="w-4 h-4" />
            مقاييس الجودة
          </button>
        </div>

        {/* Metrics row */}
        <AnimatePresence>
          {showMetrics && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="overflow-hidden border-t border-white/10"
            >
              <div className="max-w-[1600px] mx-auto px-6 py-3 grid grid-cols-2 md:grid-cols-4 gap-3">
                {metricsLoading ? (
                  <div className="col-span-4 flex items-center gap-2 text-slate-400 text-sm py-1">
                    <IconLoader2 className="w-4 h-4 animate-spin" /> جارٍ التحميل...
                  </div>
                ) : metrics ? (
                  <>
                    <MetricCard label="إجمالي الملاحظات" value={metrics.overview.totalNotes} color="purple" />
                    <MetricCard label="معدل القبول" value={`${(metrics.overview.acceptanceRate * 100).toFixed(1)}%`} color="emerald" />
                    <MetricCard label="متوسط التعديل" value={metrics.overview.avgEditDistance.toFixed(1)} color="blue" />
                    <MetricCard label="وقت المراجعة" value={`${metrics.overview.avgReviewTime.toFixed(0)}ث`} color="amber" />
                  </>
                ) : (
                  <p className="col-span-4 text-sm text-slate-500 py-1">لا توجد مقاييس بعد</p>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* ── Main 3-Column Layout ── */}
      <div className="max-w-[1600px] mx-auto px-4 py-5 grid grid-cols-1 lg:grid-cols-[290px_1fr_1fr] gap-4 items-start">

        {/* ══════════════════════════════════════════════
            LEFT PANEL — Visit Setup
            ══════════════════════════════════════════════ */}
        <div className="space-y-3">

          {/* Visit Setup Card */}
          <div className="bg-white/[0.04] border border-white/10 rounded-2xl overflow-hidden">
            <div className="flex items-center gap-2 px-4 py-3 border-b border-white/10">
              <IconUser className="w-4 h-4 text-slate-400" />
              <span className="text-sm font-semibold text-slate-200">إعداد الزيارة</span>
            </div>
            <div className="p-4 space-y-3">

              {/* Patient selector */}
              <FormField label="المريض">
                <div className="flex gap-2">
                  <select
                    value={patientId}
                    onChange={(e) => {
                      const id = e.target.value;
                      setPatientId(id);
                      const p = patients.find((pt) => pt.id === id);
                      if (p?.displayName) setPatientName(p.displayName);
                    }}
                    className="inp flex-1"
                  >
                    <option value="">— اختر مريضاً —</option>
                    {patients.map((p) => (
                      <option key={p.id} value={p.id}>{p.displayName || p.id}</option>
                    ))}
                  </select>
                  <button
                    onClick={() => setShowAddPatient((p) => !p)}
                    title={showAddPatient ? 'إلغاء' : 'مريض جديد'}
                    className="p-2.5 bg-white/5 hover:bg-white/10 border border-white/10 rounded-xl text-slate-400 hover:text-white transition-all"
                  >
                    <IconPlus className="w-4 h-4" />
                  </button>
                </div>
              </FormField>

              {/* Add patient form */}
              <AnimatePresence>
                {showAddPatient && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    className="overflow-hidden"
                  >
                    <div className="space-y-2 pt-1 pb-1">
                      <input
                        value={newPatientName}
                        onChange={(e) => setNewPatientName(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && createPatient()}
                        className="inp"
                        placeholder="اسم المريض *"
                      />
                      <input
                        value={newPatientExternalId}
                        onChange={(e) => setNewPatientExternalId(e.target.value)}
                        className="inp"
                        placeholder="رقم الملف (اختياري)"
                        dir="ltr"
                      />
                      {patientError && <p className="text-xs text-red-400">{patientError}</p>}
                      <button
                        onClick={createPatient}
                        disabled={patientStatus === 'loading'}
                        className="w-full py-2 bg-emerald-600 hover:bg-emerald-700 rounded-xl text-sm font-bold transition-colors disabled:opacity-50"
                      >
                        {patientStatus === 'loading' ? 'جارٍ الإضافة...' : 'إضافة المريض'}
                      </button>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Date of visit */}
              <FormField label="تاريخ الزيارة">
                <input
                  type="date"
                  value={dateOfVisit}
                  onChange={(e) => setDateOfVisit(e.target.value)}
                  className="inp"
                  dir="ltr"
                />
              </FormField>

              {/* Provider */}
              <FormField label="اسم الطبيب">
                <input
                  value={providerName}
                  onChange={(e) => setProviderName(e.target.value)}
                  className="inp"
                  placeholder="د. محمد أحمد"
                />
              </FormField>

              {/* Template */}
              <FormField label="قالب الملاحظة">
                <div className="flex gap-2">
                  <select
                    value={selectedTemplateId}
                    onChange={(e) => {
                      setSelectedTemplateId(e.target.value);
                      setSelectedTemplateJson(null);
                      setSelectedTemplateName('');
                    }}
                    className="inp flex-1"
                  >
                    <option value="">— اختر قالباً —</option>
                    {templates.map((t) => (
                      <option key={t.id} value={t.id}>{t.name}</option>
                    ))}
                  </select>
                  <button
                    onClick={() => setShowTemplateManager(true)}
                    title="مكتبة القوالب"
                    className="p-2.5 bg-white/5 hover:bg-white/10 border border-white/10 rounded-xl text-slate-400 hover:text-purple-300 transition-all"
                  >
                    <IconLayoutGrid className="w-4 h-4" />
                  </button>
                </div>
                {selectedTemplateName && !selectedTemplateId && (
                  <p className="text-xs text-purple-400 mt-1 flex items-center gap-1">
                    <span className="w-1.5 h-1.5 rounded-full bg-purple-400 inline-block" />
                    قالب نشط: {selectedTemplateName}
                  </p>
                )}
              </FormField>

              {/* Dialect */}
              <FormField label="لهجة التسجيل">
                <select
                  value={selectedDialect}
                  onChange={(e) => setSelectedDialect(e.target.value)}
                  className="inp"
                >
                  <option value="auto">كشف تلقائي</option>
                  <option value="egyptian">مصري</option>
                  <option value="levantine">شامي</option>
                  <option value="gulf">خليجي</option>
                  <option value="msa">فصحى</option>
                </select>
              </FormField>

            </div>
          </div>

          {/* Patient Files — collapsible */}
          <div className="bg-white/[0.04] border border-white/10 rounded-2xl overflow-hidden">
            <button
              onClick={() => setShowPatientFiles((p) => !p)}
              className="w-full flex items-center justify-between px-4 py-3 hover:bg-white/5 transition-colors"
            >
              <div className="flex items-center gap-2">
                <IconFileText className="w-4 h-4 text-slate-400" />
                <span className="text-sm font-semibold text-slate-200">ملفات المريض</span>
                {patientDocs.length > 0 && (
                  <span className="text-xs bg-blue-500/20 text-blue-300 px-1.5 py-0.5 rounded-full">
                    {patientDocs.length}
                  </span>
                )}
              </div>
              {showPatientFiles
                ? <IconChevronUp className="w-4 h-4 text-slate-500" />
                : <IconChevronDown className="w-4 h-4 text-slate-500" />
              }
            </button>
            <AnimatePresence>
              {showPatientFiles && (
                <motion.div
                  initial={{ height: 0 }}
                  animate={{ height: 'auto' }}
                  exit={{ height: 0 }}
                  className="overflow-hidden"
                >
                  <div className="px-4 pb-4">
                    {!patientId ? (
                      <p className="text-xs text-slate-500 py-2">اختر مريضاً أولاً</p>
                    ) : patientDocs.length === 0 ? (
                      <p className="text-xs text-slate-500 py-2">لا توجد ملفات لهذا المريض</p>
                    ) : (
                      <div className="space-y-2 max-h-52 overflow-y-auto custom-scrollbar">
                        {patientDocs.slice(0, 6).map((doc) => (
                          <div
                            key={doc.id}
                            className="bg-white/5 border border-white/10 rounded-xl p-3"
                          >
                            <p className="text-xs font-semibold text-slate-300 truncate">
                              {doc.title || 'وثيقة'}
                            </p>
                            {(doc.summaryText || doc.summary_text) && (
                              <p className="text-xs text-slate-500 mt-1 line-clamp-2">
                                {doc.summaryText || doc.summary_text}
                              </p>
                            )}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Session stats */}
          {recordings.length > 0 && (
            <div className="grid grid-cols-2 gap-2">
              {[
                { label: 'إجمالي', val: recordings.length, color: 'text-white' },
                { label: 'مكتمل', val: recordings.filter((r) => r.status === 'completed').length, color: 'text-emerald-300' },
                { label: 'معالجة', val: recordings.filter((r) => r.status === 'processing').length, color: 'text-yellow-300' },
                { label: 'خطأ', val: recordings.filter((r) => r.status === 'error').length, color: 'text-red-300' },
              ].map((s) => (
                <div key={s.label} className="bg-white/[0.04] border border-white/10 rounded-xl p-3 text-center">
                  <p className={`text-xl font-bold ${s.color}`}>{s.val}</p>
                  <p className="text-xs text-slate-500 mt-0.5">{s.label}</p>
                </div>
              ))}
            </div>
          )}

        </div>

        {/* ══════════════════════════════════════════════
            CENTER PANEL — Audio Input + Transcript
            ══════════════════════════════════════════════ */}
        <div className="space-y-3">

          {/* Recording Controls */}
          <div className="bg-white/[0.04] border border-white/10 rounded-2xl overflow-hidden">
            <div className="flex items-center gap-2 px-4 py-3 border-b border-white/10">
              <IconMicrophone className="w-4 h-4 text-slate-400" />
              <span className="text-sm font-semibold text-slate-200">تسجيل الجلسة</span>
            </div>
            <div className="p-4">
              <div className="flex gap-3 mb-4">
                {/* Live record button */}
                {!isRecording ? (
                  <motion.button
                    onClick={startRecording}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.97 }}
                    className="flex-1 flex flex-col items-center justify-center gap-1.5 py-5 bg-red-500/10 hover:bg-red-500/20 border border-red-500/30 rounded-2xl text-red-300 transition-all group"
                  >
                    <div className="w-10 h-10 rounded-full bg-red-500/20 group-hover:bg-red-500/30 flex items-center justify-center transition-colors">
                      <IconMicrophone className="w-5 h-5" />
                    </div>
                    <span className="text-xs font-bold">تسجيل مباشر</span>
                  </motion.button>
                ) : (
                  <motion.button
                    onClick={stopRecording}
                    animate={{ scale: [1, 1.02, 1] }}
                    transition={{ repeat: Infinity, duration: 1.2 }}
                    className="flex-1 flex flex-col items-center justify-center gap-1.5 py-5 bg-red-500/20 border border-red-500/50 rounded-2xl text-red-300"
                  >
                    <div className="w-10 h-10 rounded-full bg-red-500/30 flex items-center justify-center">
                      <IconPlayerStop className="w-5 h-5" />
                    </div>
                    <span className="text-sm font-mono font-bold">{formatTime(recordingTime)}</span>
                    <span className="text-xs opacity-70">اضغط للإيقاف</span>
                  </motion.button>
                )}

                {/* Upload button */}
                <motion.button
                  onClick={() => fileInputRef.current?.click()}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.97 }}
                  className="flex-1 flex flex-col items-center justify-center gap-1.5 py-5 bg-blue-500/10 hover:bg-blue-500/20 border border-blue-500/30 rounded-2xl text-blue-300 transition-all group"
                >
                  <div className="w-10 h-10 rounded-full bg-blue-500/20 group-hover:bg-blue-500/30 flex items-center justify-center transition-colors">
                    <IconUpload className="w-5 h-5" />
                  </div>
                  <span className="text-xs font-bold">رفع ملف</span>
                  <span className="text-xs opacity-50">MP3 WAV M4A</span>
                </motion.button>
                <input ref={fileInputRef} type="file" accept="audio/*" multiple onChange={handleFileUpload} className="hidden" />
              </div>

              {/* Recording list */}
              {recordings.length === 0 ? (
                <div className="text-center py-8 border border-dashed border-white/10 rounded-xl">
                  <IconMicrophone className="w-8 h-8 text-slate-700 mx-auto mb-2" />
                  <p className="text-sm text-slate-500">سجِّل أو ارفع ملفاً للبدء</p>
                  <p className="text-xs text-slate-600 mt-1">تأكد من اختيار المريض أولاً</p>
                </div>
              ) : (
                <div className="space-y-2 max-h-72 overflow-y-auto custom-scrollbar">
                  {recordings.map((rec) => (
                    <button
                      key={rec.id}
                      onClick={() => setSelectedRecording(rec)}
                      className={`w-full text-right flex items-center gap-3 px-3 py-2.5 rounded-xl border transition-all ${
                        selectedRecording?.id === rec.id
                          ? 'border-purple-500/60 bg-purple-500/10'
                          : 'border-white/10 bg-white/[0.03] hover:bg-white/[0.06] hover:border-white/20'
                      }`}
                    >
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-white truncate">{rec.file.name}</p>
                        <p className="text-xs text-slate-500">{rec.timestamp.toLocaleString('ar-EG')}</p>
                      </div>
                      <StatusBadge status={rec.status} />
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Transcript panel */}
          <AnimatePresence mode="wait">
            {selectedRecording && (
              <motion.div
                key={selectedRecording.id}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                className="bg-white/[0.04] border border-white/10 rounded-2xl overflow-hidden"
              >
                <div className="flex items-center gap-2 px-4 py-3 border-b border-white/10">
                  <IconFileText className="w-4 h-4 text-slate-400" />
                  <span className="text-sm font-semibold text-slate-200">النص المفرَّغ</span>
                  {selectedRecording.dialect && selectedRecording.dialect !== 'auto' && (
                    <span className="text-xs bg-white/10 text-slate-400 px-2 py-0.5 rounded-full mr-auto">
                      {selectedRecording.dialect === 'egyptian' ? 'مصري'
                        : selectedRecording.dialect === 'levantine' ? 'شامي'
                        : selectedRecording.dialect === 'gulf' ? 'خليجي'
                        : selectedRecording.dialect}
                    </span>
                  )}
                </div>
                <div className="p-4">
                  {selectedRecording.status === 'processing' ? (
                    <div className="flex items-center gap-3 py-6 justify-center text-slate-400">
                      <IconLoader2 className="w-5 h-5 animate-spin" />
                      <span className="text-sm">جارٍ التفريغ والتحليل...</span>
                    </div>
                  ) : selectedRecording.status === 'error' ? (
                    <div className="flex items-start gap-2 bg-red-500/10 border border-red-500/30 rounded-xl p-3">
                      <IconX className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
                      <p className="text-sm text-red-300">{selectedRecording.error}</p>
                    </div>
                  ) : selectedRecording.transcript ? (
                    <div className="bg-slate-800/50 border border-white/[0.08] rounded-xl p-4 max-h-60 overflow-y-auto custom-scrollbar">
                      <p className="text-sm text-slate-300 leading-loose whitespace-pre-wrap">
                        {selectedRecording.transcript}
                      </p>
                    </div>
                  ) : (
                    <p className="text-sm text-slate-500 text-center py-4">لا يوجد نص مفرَّغ</p>
                  )}
                </div>
              </motion.div>
            )}
          </AnimatePresence>

        </div>

        {/* ══════════════════════════════════════════════
            RIGHT PANEL — SOAP Editor
            ══════════════════════════════════════════════ */}
        <div className="space-y-3">

          {!hasNote ? (
            // Empty state
            <div className="bg-white/[0.04] border border-white/10 rounded-2xl p-8 text-center">
              <div className="w-16 h-16 rounded-2xl bg-white/5 flex items-center justify-center mx-auto mb-4">
                <IconFileText className="w-8 h-8 text-slate-600" />
              </div>
              <p className="text-base font-semibold text-slate-300 mb-2">
                ملاحظة SOAP
              </p>
              <p className="text-sm text-slate-500 leading-relaxed">
                سجِّل أو ارفع ملفاً صوتياً لجلسة المريض<br />
                وسيقوم النظام بتوليد الملاحظة تلقائياً
              </p>
              {!patientId && (
                <p className="text-xs text-amber-400 mt-3 bg-amber-400/10 border border-amber-400/20 rounded-lg px-3 py-2">
                  ⚠ اختر مريضاً من الإعدادات أولاً
                </p>
              )}
            </div>
          ) : (
            <>
              {/* SOAP section header */}
              <div className="flex items-center justify-between px-1">
                <span className="text-sm font-semibold text-slate-300">ملاحظة SOAP</span>
                {selectedRecording?.noteId && (
                  <span className="text-xs text-slate-500">
                    انقر على 🎤 بجانب أي قسم لتحديثه صوتياً
                  </span>
                )}
              </div>

              {/* 4 SOAP sections */}
              {SOAP_SECTIONS.map((section) => (
                <SoapSectionCard
                  key={section.key}
                  section={section}
                  value={soapSections[section.key]}
                  onChange={(v) => {
                    setSoapSections((p) => ({ ...p, [section.key]: v }));
                    if (selectedRecording && !selectedRecording.reviewStartTime) {
                      setSelectedRecording((p) => p ? { ...p, reviewStartTime: Date.now() } : p);
                    }
                  }}
                  isRecording={sectionRecording === section.key}
                  isUpdating={sectionUpdating === section.key}
                  recordingTime={sectionRecording === section.key ? sectionRecordingTime : 0}
                  onStartRecording={() => startSectionRecording(section.key)}
                  onStopRecording={stopSectionRecording}
                  canRecord={!!selectedRecording?.noteId}
                />
              ))}

              {/* Action bar */}
              <div className="flex gap-2 pt-1">
                <motion.button
                  onClick={saveToEHR}
                  disabled={isSaving || !canSave}
                  whileHover={{ scale: canSave ? 1.01 : 1 }}
                  whileTap={{ scale: canSave ? 0.99 : 1 }}
                  className="flex-1 flex items-center justify-center gap-2 py-3 bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-700 disabled:text-slate-500 disabled:cursor-not-allowed rounded-xl font-bold text-sm transition-colors shadow-lg shadow-emerald-900/40"
                >
                  {isSaving
                    ? <><IconLoader2 className="w-4 h-4 animate-spin" /> جارٍ الحفظ...</>
                    : <><IconCheck className="w-4 h-4" /> قبول وإرسال للسجل</>
                  }
                </motion.button>

                <button
                  onClick={rejectNote}
                  title="رفض الملاحظة"
                  className="px-4 py-3 bg-red-500/10 hover:bg-red-500/20 border border-red-500/30 rounded-xl text-red-400 hover:text-red-300 transition-colors"
                >
                  <IconX className="w-4 h-4" />
                </button>

                <button
                  onClick={downloadNote}
                  title="تنزيل"
                  className="px-4 py-3 bg-blue-500/10 hover:bg-blue-500/20 border border-blue-500/30 rounded-xl text-blue-400 hover:text-blue-300 transition-colors"
                >
                  <IconFileDownload className="w-4 h-4" />
                </button>
              </div>
            </>
          )}

        </div>
      </div>

      {/* ── Template Manager Modal ── */}
      <AnimatePresence>
        {showTemplateManager && (
          <TemplateManagerModal
            tab={templateTab}
            onTabChange={setTemplateTab}
            templates={templates}
            builtinTemplates={BUILTIN_TEMPLATES}
            customTemplateName={customTemplateName}
            customTemplateText={customTemplateText}
            templateSaving={templateSaving}
            templateError={templateError}
            templateFileInputRef={templateFileInputRef}
            onClose={() => { setShowTemplateManager(false); setTemplateError(''); }}
            onNameChange={setCustomTemplateName}
            onTextChange={setCustomTemplateText}
            onFileUpload={handleTemplateFileUpload}
            onSaveCustom={async () => {
              if (!customTemplateName.trim()) { setTemplateError('أدخل اسم القالب'); return; }
              try {
                const json = JSON.parse(customTemplateText);
                await createCustomTemplate(customTemplateName.trim(), json);
              } catch {
                setTemplateError('تنسيق JSON غير صحيح. تحقق من البنية.');
              }
            }}
            onSelectBuiltin={(t) => {
              setSelectedTemplateJson(t.template as Record<string, any>);
              setSelectedTemplateName(t.name);
              setSelectedTemplateId('');
              setShowTemplateManager(false);
              showToast('success', `تم تفعيل قالب: ${t.name}`);
            }}
            onSaveBuiltin={(t) => createCustomTemplate(t.name, t.template as Record<string, any>)}
          />
        )}
      </AnimatePresence>

      {/* ── Toast ── */}
      <AnimatePresence>
        {toast && (
          <motion.div
            initial={{ opacity: 0, y: 40, x: '-50%' }}
            animate={{ opacity: 1, y: 0, x: '-50%' }}
            exit={{ opacity: 0, y: 40, x: '-50%' }}
            className={`fixed bottom-6 left-1/2 z-50 flex items-center gap-2 px-5 py-3 rounded-xl shadow-2xl text-sm font-bold ${
              toast.type === 'success'
                ? 'bg-emerald-600 shadow-emerald-900/60'
                : 'bg-red-600 shadow-red-900/60'
            }`}
          >
            {toast.type === 'success'
              ? <IconCheck className="w-4 h-4 shrink-0" />
              : <IconX className="w-4 h-4 shrink-0" />
            }
            {toast.message}
          </motion.div>
        )}
      </AnimatePresence>

      <style>{`
        .inp {
          width: 100%;
          padding: 0.5rem 0.75rem;
          background: rgba(255,255,255,0.05);
          border: 1px solid rgba(255,255,255,0.1);
          border-radius: 0.75rem;
          color: white;
          font-size: 0.875rem;
          font-family: 'Tajawal', sans-serif;
          transition: all 0.15s;
          outline: none;
        }
        .inp:focus {
          border-color: rgba(168,85,247,0.5);
          box-shadow: 0 0 0 2px rgba(168,85,247,0.2);
        }
        .inp option { background: #1e293b; color: white; }
        .custom-scrollbar::-webkit-scrollbar { width: 4px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.12); border-radius: 4px; }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }
      `}</style>
    </div>
  );
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function FormField({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="block text-xs font-medium text-slate-400 mb-1.5">{label}</label>
      {children}
    </div>
  );
}

function StatusBadge({ status }: { status: AudioRecording['status'] }) {
  if (status === 'completed')
    return (
      <span className="shrink-0 text-[11px] px-2 py-0.5 bg-emerald-500/15 text-emerald-300 border border-emerald-500/30 rounded-full flex items-center gap-1">
        <IconCheck className="w-3 h-3" />مكتمل
      </span>
    );
  if (status === 'processing')
    return (
      <span className="shrink-0 text-[11px] px-2 py-0.5 bg-yellow-500/15 text-yellow-300 border border-yellow-500/30 rounded-full flex items-center gap-1">
        <IconLoader2 className="w-3 h-3 animate-spin" />معالجة
      </span>
    );
  if (status === 'error')
    return (
      <span className="shrink-0 text-[11px] px-2 py-0.5 bg-red-500/15 text-red-300 border border-red-500/30 rounded-full flex items-center gap-1">
        <IconX className="w-3 h-3" />خطأ
      </span>
    );
  return (
    <span className="shrink-0 text-[11px] px-2 py-0.5 bg-slate-500/15 text-slate-400 border border-slate-500/30 rounded-full flex items-center gap-1">
      <IconClock className="w-3 h-3" />انتظار
    </span>
  );
}

function MetricCard({
  label, value, color,
}: { label: string; value: string | number; color: string }) {
  const palette: Record<string, string> = {
    purple: 'text-purple-300',
    emerald: 'text-emerald-300',
    blue: 'text-blue-300',
    amber: 'text-amber-300',
  };
  return (
    <div className="bg-white/[0.04] border border-white/10 rounded-xl p-3">
      <p className="text-xs text-slate-500 mb-0.5">{label}</p>
      <p className={`text-2xl font-bold ${palette[color] ?? 'text-white'}`}>{value}</p>
    </div>
  );
}

function SoapSectionCard({
  section,
  value,
  onChange,
  isRecording,
  isUpdating,
  recordingTime,
  onStartRecording,
  onStopRecording,
  canRecord,
}: {
  section: typeof SOAP_SECTIONS[number];
  value: string;
  onChange: (v: string) => void;
  isRecording: boolean;
  isUpdating: boolean;
  recordingTime: number;
  onStartRecording: () => void;
  onStopRecording: () => void;
  canRecord: boolean;
}) {
  return (
    <div className={`border ${section.border} rounded-2xl overflow-hidden`}>
      {/* Section header */}
      <div className={`${section.header} flex items-center justify-between px-4 py-2.5`}>
        <div className="flex items-center gap-2">
          <span className={`w-2 h-2 rounded-full ${section.dot}`} />
          <span className={`text-sm font-bold ${section.badge}`}>{section.label}</span>
          <span className="text-xs text-slate-500 hidden sm:block">{section.desc}</span>
        </div>

        {/* Voice mic per section */}
        {canRecord && (
          <div className="flex items-center gap-2">
            {isUpdating && (
              <span className="text-xs text-slate-400 flex items-center gap-1">
                <IconLoader2 className="w-3 h-3 animate-spin" />
                يعالج...
              </span>
            )}
            {!isRecording ? (
              <button
                onClick={onStartRecording}
                disabled={isUpdating}
                title="تحديث هذا القسم بالصوت"
                className={`p-1.5 rounded-lg hover:bg-white/20 transition-all disabled:opacity-30 ${section.badge} bg-white/10`}
              >
                <IconMicrophone className="w-3.5 h-3.5" />
              </button>
            ) : (
              <button
                onClick={onStopRecording}
                className="flex items-center gap-1 px-2.5 py-1 rounded-lg bg-red-500/25 border border-red-500/40 text-red-300 text-xs font-mono animate-pulse"
              >
                <IconPlayerStop className="w-3 h-3" />
                {formatTime(recordingTime)}
              </button>
            )}
          </div>
        )}
      </div>

      {/* Section textarea */}
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className={`w-full px-4 py-3 ${section.bg} text-sm text-slate-200 leading-loose resize-none focus:outline-none focus:ring-2 ${section.ring} focus:ring-inset`}
        rows={4}
        dir="rtl"
        placeholder={section.placeholder}
      />

      {/* Word count */}
      {value && (
        <div className={`${section.bg} px-4 pb-2 text-right`}>
          <span className="text-[11px] text-slate-600">
            {value.trim().split(/\s+/).filter(Boolean).length} كلمة
          </span>
        </div>
      )}
    </div>
  );
}

// ─── Template Manager Modal ────────────────────────────────────────────────────

const COLOR_MAP: Record<string, { bg: string; text: string; border: string }> = {
  purple:  { bg: 'bg-purple-500/10',  text: 'text-purple-300',  border: 'border-purple-500/30' },
  blue:    { bg: 'bg-blue-500/10',    text: 'text-blue-300',    border: 'border-blue-500/30' },
  indigo:  { bg: 'bg-indigo-500/10',  text: 'text-indigo-300',  border: 'border-indigo-500/30' },
  emerald: { bg: 'bg-emerald-500/10', text: 'text-emerald-300', border: 'border-emerald-500/30' },
  red:     { bg: 'bg-red-500/10',     text: 'text-red-300',     border: 'border-red-500/30' },
  amber:   { bg: 'bg-amber-500/10',   text: 'text-amber-300',   border: 'border-amber-500/30' },
  teal:    { bg: 'bg-teal-500/10',    text: 'text-teal-300',    border: 'border-teal-500/30' },
  cyan:    { bg: 'bg-cyan-500/10',    text: 'text-cyan-300',    border: 'border-cyan-500/30' },
  rose:    { bg: 'bg-rose-500/10',    text: 'text-rose-300',    border: 'border-rose-500/30' },
  orange:  { bg: 'bg-orange-500/10',  text: 'text-orange-300',  border: 'border-orange-500/30' },
  pink:    { bg: 'bg-pink-500/10',    text: 'text-pink-300',    border: 'border-pink-500/30' },
  slate:   { bg: 'bg-slate-500/10',   text: 'text-slate-300',   border: 'border-slate-500/30' },
};

function TemplateManagerModal({
  tab, onTabChange, templates, builtinTemplates,
  customTemplateName, customTemplateText,
  templateSaving, templateError, templateFileInputRef,
  onClose, onNameChange, onTextChange, onFileUpload,
  onSaveCustom, onSelectBuiltin, onSaveBuiltin,
}: {
  tab: 'builtin' | 'upload' | 'custom';
  onTabChange: (t: 'builtin' | 'upload' | 'custom') => void;
  templates: Array<{ id: string; name: string }>;
  builtinTemplates: typeof BUILTIN_TEMPLATES;
  customTemplateName: string;
  customTemplateText: string;
  templateSaving: boolean;
  templateError: string;
  templateFileInputRef: React.RefObject<HTMLInputElement>;
  onClose: () => void;
  onNameChange: (v: string) => void;
  onTextChange: (v: string) => void;
  onFileUpload: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onSaveCustom: () => void;
  onSelectBuiltin: (t: typeof BUILTIN_TEMPLATES[number]) => void;
  onSaveBuiltin: (t: typeof BUILTIN_TEMPLATES[number]) => Promise<void>;
}) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4"
      dir="rtl"
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <motion.div
        initial={{ opacity: 0, scale: 0.95, y: 16 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.95, y: 16 }}
        className="bg-slate-900 border border-white/15 rounded-2xl w-full max-w-3xl max-h-[88vh] flex flex-col shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-white/10">
          <div>
            <h2 className="text-lg font-bold text-white">مكتبة القوالب</h2>
            <p className="text-xs text-slate-500 mt-0.5">اختر قالباً جاهزاً أو أنشئ قالبك الخاص</p>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-white/10 rounded-xl text-slate-400 hover:text-white transition-colors"
          >
            <IconX className="w-5 h-5" />
          </button>
        </div>

        {/* Tabs */}
        <div className="flex gap-2 px-6 pt-4">
          {([
            { key: 'builtin' as const, label: 'قوالب جاهزة',  icon: <IconLayoutGrid className="w-4 h-4" /> },
            { key: 'upload'  as const, label: 'رفع ملف JSON', icon: <IconUpload className="w-4 h-4" /> },
            { key: 'custom'  as const, label: 'قالب مخصص',    icon: <IconCode className="w-4 h-4" /> },
          ]).map((t) => (
            <button
              key={t.key}
              onClick={() => onTabChange(t.key)}
              className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all ${
                tab === t.key
                  ? 'bg-purple-500/20 border border-purple-500/40 text-purple-200'
                  : 'bg-white/5 border border-white/10 text-slate-400 hover:bg-white/10 hover:text-white'
              }`}
            >
              {t.icon} {t.label}
            </button>
          ))}
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto p-6 custom-scrollbar">

          {/* ── Tab: Built-in ── */}
          {tab === 'builtin' && (
            <div>
              <p className="text-xs text-slate-500 mb-4">
                اضغط <strong className="text-slate-300">تفعيل</strong> لاستخدام القالب فوراً أو
                <strong className="text-slate-300"> حفظ</strong> لإضافته لقائمتك الدائمة
              </p>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                {builtinTemplates.map((t) => {
                  const c = COLOR_MAP[t.color] ?? COLOR_MAP.slate;
                  return (
                    <div key={t.id} className={`${c.bg} ${c.border} border rounded-xl p-4 flex flex-col gap-2`}>
                      <span className="text-2xl">{t.icon}</span>
                      <div>
                        <p className={`text-sm font-bold ${c.text}`}>{t.name}</p>
                        <p className="text-[11px] text-slate-500 leading-tight mt-0.5">{t.desc}</p>
                        <p className="text-[10px] text-slate-600 mt-0.5 font-mono">{t.nameEn}</p>
                      </div>
                      <div className="flex gap-2 mt-auto">
                        <button
                          onClick={() => onSelectBuiltin(t)}
                          className={`flex-1 py-1.5 text-xs font-bold rounded-lg border ${c.bg} ${c.border} ${c.text} hover:opacity-80 transition-opacity`}
                        >
                          ⚡ تفعيل
                        </button>
                        <button
                          onClick={() => onSaveBuiltin(t)}
                          className="flex-1 py-1.5 text-xs font-bold rounded-lg bg-white/5 border border-white/10 text-slate-400 hover:bg-white/10 hover:text-white transition-all"
                        >
                          💾 حفظ
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* ── Tab: Upload ── */}
          {tab === 'upload' && (
            <div className="max-w-md mx-auto space-y-4">
              <div className="bg-blue-500/10 border border-blue-500/20 rounded-xl p-4">
                <p className="text-sm text-blue-200 font-semibold mb-1">كيفية رفع قالب</p>
                <p className="text-xs text-slate-400 leading-relaxed">
                  ارفع ملف <code className="text-blue-300 font-mono">.json</code> يحتوي على بنية القالب.
                  يمكنك تصدير قالب موجود من أي نظام EHR أو إنشاؤه يدوياً.
                </p>
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1.5">اسم القالب (اختياري)</label>
                <input
                  value={customTemplateName}
                  onChange={(e) => onNameChange(e.target.value)}
                  className="inp"
                  placeholder="سيُستخدم اسم الملف إن تُرك فارغاً"
                />
              </div>
              <button
                onClick={() => templateFileInputRef.current?.click()}
                className="w-full flex flex-col items-center gap-3 py-10 border-2 border-dashed border-white/20 hover:border-purple-500/50 rounded-xl text-slate-400 hover:text-white transition-all group"
              >
                <div className="w-12 h-12 rounded-full bg-white/5 group-hover:bg-purple-500/10 flex items-center justify-center transition-colors">
                  <IconUpload className="w-6 h-6" />
                </div>
                <div className="text-center">
                  <p className="text-sm font-bold">اضغط لاختيار ملف JSON</p>
                  <p className="text-xs opacity-60 mt-0.5">الحد الأقصى 1 MB</p>
                </div>
              </button>
              <input
                ref={templateFileInputRef}
                type="file"
                accept=".json,application/json"
                className="hidden"
                onChange={onFileUpload}
              />
              {templateError && (
                <p className="text-sm text-red-400 bg-red-500/10 border border-red-500/20 rounded-xl px-3 py-2">{templateError}</p>
              )}
            </div>
          )}

          {/* ── Tab: Custom ── */}
          {tab === 'custom' && (
            <div className="space-y-4">
              <div className="bg-purple-500/10 border border-purple-500/20 rounded-xl p-4">
                <p className="text-sm text-purple-200 font-semibold mb-1">إنشاء قالب مخصص</p>
                <p className="text-xs text-slate-400 leading-relaxed">
                  عرِّف قالبك بتنسيق JSON. اضغط <strong className="text-slate-300">استيراد مثال</strong> للبدء.
                  الأقسام المتاحة: <code className="text-emerald-300 font-mono">subjective, objective, assessment, plan, data, behavior, intervention, response</code>
                </p>
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1.5">اسم القالب *</label>
                <input
                  value={customTemplateName}
                  onChange={(e) => onNameChange(e.target.value)}
                  className="inp"
                  placeholder="مثال: قالب طب الأسرة"
                />
              </div>
              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <label className="block text-xs font-medium text-slate-400">بنية القالب (JSON)</label>
                  <button
                    onClick={() => onTextChange('{\n  "style": "standard",\n  "sections": ["subjective", "objective", "assessment", "plan"],\n  "prompts": {\n    "subjective": "شكوى المريض الرئيسية...",\n    "objective": "الفحص السريري والعلامات الحيوية...",\n    "assessment": "التشخيص والتقييم السريري...",\n    "plan": "الخطة العلاجية والمتابعة..."\n  }\n}')}
                    className="text-xs text-purple-400 hover:text-purple-300 transition-colors"
                  >
                    <IconEdit className="w-3 h-3 inline ml-1" />
                    استيراد مثال
                  </button>
                </div>
                <textarea
                  value={customTemplateText}
                  onChange={(e) => onTextChange(e.target.value)}
                  className="w-full h-56 px-4 py-3 bg-slate-800/60 border border-white/10 rounded-xl text-sm text-emerald-300 leading-relaxed resize-none focus:outline-none focus:border-purple-500/50 focus:ring-2 focus:ring-purple-500/20 font-mono"
                  placeholder={'{}  '}
                  dir="ltr"
                  spellCheck={false}
                />
              </div>
              {templateError && (
                <p className="text-sm text-red-400 bg-red-500/10 border border-red-500/20 rounded-xl px-3 py-2">{templateError}</p>
              )}
              <button
                onClick={onSaveCustom}
                disabled={templateSaving}
                className="w-full py-3 bg-purple-600 hover:bg-purple-700 disabled:bg-slate-700 disabled:text-slate-500 rounded-xl font-bold text-sm transition-colors flex items-center justify-center gap-2"
              >
                {templateSaving
                  ? <><IconLoader2 className="w-4 h-4 animate-spin" /> جارٍ الحفظ...</>
                  : <><IconCheck className="w-4 h-4" /> حفظ القالب</>
                }
              </button>
            </div>
          )}
        </div>

        {/* Footer: saved templates */}
        {templates.length > 0 && (
          <div className="px-6 py-3 border-t border-white/10">
            <p className="text-xs text-slate-500">
              قوالبك المحفوظة: {templates.map((t) => t.name).join(' • ')}
            </p>
          </div>
        )}
      </motion.div>
    </div>
  );
}
