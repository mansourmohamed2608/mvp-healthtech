import { useState, useEffect, useRef, useCallback } from 'react';
import { useThemeStore } from '@store/themeStore';
import { useAuthStore } from '@store/authStore';
import {
  IconPhone,
  IconPhoneOff,
  IconMicrophone,
  IconMicrophoneOff,
  IconCheck,
  IconAlertCircle,
  IconLoader2,
  IconSettings,
  IconEar,
  IconVolume,
  IconMessage,
  IconSend,
  IconPlayerPlay,
} from '@tabler/icons-react';
import { Device, Call } from '@twilio/voice-sdk';
import api from '../utils/api';
import clsx from 'clsx';

interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
}

type VoiceActivityState = 'idle' | 'listening' | 'processing' | 'speaking';

const VoiceAgentClean = () => {
  const { theme, language } = useThemeStore();
  const { token } = useAuthStore();
  const [device, setDevice] = useState<Device | null>(null);
  const [call, setCall] = useState<Call | null>(null);
  const [callStatus, setCallStatus] = useState<'idle' | 'connecting' | 'connected' | 'disconnected'>('idle');
  const [transcript, setTranscript] = useState<Message[]>([]);
  const [isThinking, setIsThinking] = useState(false);
  const [error, setError] = useState<string>('');
  const [isMuted, setIsMuted] = useState(false);
  const [deviceReady, setDeviceReady] = useState(false);
  const [callSid, setCallSid] = useState<string>('');
  const [dialectPreference, setDialectPreference] = useState<'auto' | 'egypt' | 'saudi'>('saudi');
  const [voicePreference, setVoicePreference] = useState<'auto' | 'egypt' | 'saudi'>('saudi');
  const [prefsStatus, setPrefsStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle');
  const [prefsError, setPrefsError] = useState<string>('');
  const [introMessage, setIntroMessage] = useState<Message | null>(null);
  const [voiceActivity, setVoiceActivity] = useState<VoiceActivityState>('idle');
  const [showSettings, setShowSettings] = useState(false);
  const transcriptEndRef = useRef<HTMLDivElement>(null);
  const lastMessageRef = useRef<Message | null>(null);

  // ── Scripted Call Demo ────────────────────────────────────────────────────
  const [demoStep, setDemoStep] = useState(-1);
  const [demoPhase, setDemoPhase] = useState<'idle' | 'va' | 'listening' | 'done'>('idle');
  const [demoConvo, setDemoConvo] = useState<{ role: 'va' | 'patient'; text: string }[]>([]);
  const [demoTypeText, setDemoTypeText] = useState('');
  const [callSeconds, setCallSeconds] = useState(0);
  const demoAbortRef = useRef(false);
  const demoEndRef = useRef<HTMLDivElement>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const callTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Fully hardcoded demo script ─────────────────────────────────────────
  // Both VA lines and patient cues are fixed.  No LLM call is made.
  const DEMO_SCRIPT = [
    {
      va: 'أهلاً وسهلاً! أنا ليان، مساعدة مستشفى بيميدكس الذكية. كيف أقدر أساعدك اليوم؟',
      patient: 'أبغى أحجز موعد في قسم الجلدية',
    },
    {
      va: 'بكل سرور! سأحجز لك موعد في قسم الجلدية. ما اسمك الكريم؟',
      patient: 'اسمي منصور محمد منصور',
    },
    {
      va: 'شكراً منصور. ما رقم جوالك من فضلك؟',
      patient: '٠١٠٩٥٠١٣٥٣٦',
    },
    {
      va: 'تمام. وما تاريخ ميلادك؟',
      patient: '٢٦ / ٠٨ / ٢٠٠١',
    },
    {
      va: 'ممتاز. أي يوم يناسبك للموعد؟',
      patient: 'الثلاثاء',
    },
    {
      va: 'الثلاثاء متاح. أي وقت يناسبك؟',
      patient: 'الساعة الثالثة مساءً',
    },
    {
      va: 'تم تأكيد موعدك يوم الثلاثاء الساعة الثالثة مساءً في قسم الجلدية. شكراً لثقتك بمستشفى بيميدكس يا منصور!',
      patient: null,
    },
  ] as const;
  // How long the green listening bar stays per step (ms)
  const LISTEN_MS = 3500;

  // G.711 μ-law → 16-bit PCM decoder — matches Python audioop.ulaw2lin exactly
  function mulawDecode(byte: number): number {
    // Lookup table matches CPython audioop exp_lut[8]
    const expLut = [0, 132, 396, 924, 1980, 4092, 8316, 16764];
    byte = ~byte & 0xff;
    const sign  = byte & 0x80;
    const exp   = (byte >> 4) & 0x07;
    const mant  = byte & 0x0f;
    const sample = expLut[exp] + (mant << (exp + 3));
    return sign ? sample : -sample;
  }

  const playMulawAudio = useCallback(async (base64: string, sampleRate = 8000) => {
    if (!base64) return;
    try {
      if (!audioCtxRef.current || audioCtxRef.current.state === 'closed') {
        audioCtxRef.current = new AudioContext();
      }
      const ctx = audioCtxRef.current;
      if (ctx.state === 'suspended') await ctx.resume();
      const raw = atob(base64);
      const samples = new Float32Array(raw.length);
      for (let i = 0; i < raw.length; i++) {
        samples[i] = mulawDecode(raw.charCodeAt(i)) / 32768.0;
      }
      const buf = ctx.createBuffer(1, samples.length, sampleRate);
      buf.copyToChannel(samples, 0);
      const src = ctx.createBufferSource();
      src.buffer = buf;
      src.connect(ctx.destination);
      src.start();
      await new Promise<void>(r => { src.onended = () => r(); });
    } catch { /* non-fatal */ }
  }, []);

  // Auto-scroll conversation panel
  useEffect(() => {
    demoEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [demoConvo, demoTypeText]);

  // Typewriter helper
  const typewrite = useCallback(async (text: string, abortRef: React.MutableRefObject<boolean>) => {
    for (let c = 1; c <= text.length; c++) {
      if (abortRef.current) return;
      setDemoTypeText(text.slice(0, c));
      await new Promise(r => setTimeout(r, 36));
    }
  }, []);

  // ── Fully hardcoded scripted demo — looks like a real call ───────────────
  const runScriptedDemo = useCallback(async () => {
    demoAbortRef.current = false;
    setDemoConvo([]);
    setDemoTypeText('');
    setDemoStep(-1);
    setDemoPhase('idle');

    // Connecting phase
    setCallStatus('connecting');
    await new Promise(r => setTimeout(r, 1800));
    if (demoAbortRef.current) { setCallStatus('idle'); return; }
    setCallStatus('connected');
    await new Promise(r => setTimeout(r, 400));

    for (let i = 0; i < DEMO_SCRIPT.length; i++) {
      if (demoAbortRef.current) break;
      const step = DEMO_SCRIPT[i];
      setDemoStep(i);

      // VA speaks: typewrite + TTS in parallel
      setDemoPhase('va');
      setDemoTypeText('');
      const ttsPromise = api.synthesizeSpeech(step.va, 'saudi-tts').catch(() => null);
      await typewrite(step.va, demoAbortRef);
      if (demoAbortRef.current) break;

      setDemoConvo(prev => [...prev, { role: 'va', text: step.va }]);
      setDemoTypeText('');

      // Play Ahmed Saudi TTS
      try {
        const ttsRes = await ttsPromise;
        if (ttsRes && (ttsRes as any).audio) {
          await playMulawAudio((ttsRes as any).audio, (ttsRes as any).sampleRate || 8000);
        }
      } catch { /* non-fatal */ }
      if (demoAbortRef.current) break;

      // Patient turn: 2 second pause then auto-advance
      if (step.patient) {
        setDemoPhase('listening');
        await new Promise(r => setTimeout(r, 2000));
        if (demoAbortRef.current) break;
        setDemoConvo(prev => [...prev, { role: 'patient', text: step.patient as string }]);
        await new Promise(r => setTimeout(r, 400));
      }
      if (demoAbortRef.current) break;
    }

    if (!demoAbortRef.current) {
      setDemoPhase('done');
      setDemoTypeText('');
      await new Promise(r => setTimeout(r, 2500));
      if (!demoAbortRef.current) {
        setCallStatus('disconnected');
        await new Promise(r => setTimeout(r, 2500));
        if (!demoAbortRef.current) {
          setCallStatus('idle');
          setDemoStep(-1);
        }
      }
    }
  }, [playMulawAudio, typewrite]);

  const stopDemo = useCallback(() => {
    demoAbortRef.current = true;
    setDemoPhase('idle');
    setDemoTypeText('');
  }, []);

  // Start call: create AudioContext in user-gesture context then launch demo
  const handleCallButton = useCallback(async () => {
    try {
      if (!audioCtxRef.current || audioCtxRef.current.state === 'closed') {
        audioCtxRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
      }
      await audioCtxRef.current.resume();
    } catch { /* non-fatal */ }
    runScriptedDemo();
  }, [runScriptedDemo]);

  // End call button
  const handleEndCall = useCallback(() => {
    demoAbortRef.current = true;
    setDemoPhase('idle');
    setDemoTypeText('');
    setCallStatus('disconnected');
    setTimeout(() => {
      setCallStatus('idle');
      setDemoStep(-1);
    }, 2000);
  }, []);

  // Call timer
  useEffect(() => {
    if (callStatus === 'connected') {
      setCallSeconds(0);
      callTimerRef.current = setInterval(() => setCallSeconds(s => s + 1), 1000);
    } else {
      if (callTimerRef.current) { clearInterval(callTimerRef.current); callTimerRef.current = null; }
      if (callStatus === 'idle') setCallSeconds(0);
    }
    return () => { if (callTimerRef.current) { clearInterval(callTimerRef.current); callTimerRef.current = null; } };
  }, [callStatus]);

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const resolveVoiceId = (pref: string) => {
    if (pref === 'egypt') return 'egtts';
    if (pref === 'saudi') return 'saudi-tts';
    return undefined;
  };

  // Auto-scroll transcript
  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [transcript]);

  // Sync voice preferences
  useEffect(() => {
    if (!callSid || callStatus !== 'connected') return;
    let cancelled = false;

    const syncPreferences = async () => {
      setPrefsStatus('saving');
      setPrefsError('');
      try {
        const voicePayload = voicePreference === 'auto' ? 'auto' : resolveVoiceId(voicePreference);
        await api.updateConversationPreferences(callSid, {
          dialect: dialectPreference,
          voice: voicePayload,
        });
        if (!cancelled) setPrefsStatus('saved');
      } catch (err: any) {
        if (!cancelled) {
          setPrefsStatus('error');
          setPrefsError(err.message || 'Failed to update preferences');
        }
      }
    };

    syncPreferences();
    return () => { cancelled = true; };
  }, [callSid, callStatus, dialectPreference, voicePreference]);

  // Poll transcript history
  useEffect(() => {
    if (!callSid || callStatus !== 'connected') return;
    let cancelled = false;

    const fetchHistory = async () => {
      try {
        const data = await api.getConversationHistory(callSid, 80);
        if (cancelled) return;
        const historyMessages: Message[] = (data.messages || []).map((msg: any) => ({
          role: msg.role === 'assistant' || msg.role === 'system' ? msg.role : 'user',
          content: msg.content,
          timestamp: new Date(msg.timestamp),
        }));
        const merged = introMessage ? [introMessage, ...historyMessages] : historyMessages;
        setTranscript(merged);
        
        const last = historyMessages[historyMessages.length - 1];
        const isNewMessage = last && (!lastMessageRef.current || 
          lastMessageRef.current.content !== last.content);
        
        if (last) {
          lastMessageRef.current = last;
          if (last.role === 'user') {
            setVoiceActivity('processing');
            setIsThinking(true);
          } else if (last.role === 'assistant') {
            if (isNewMessage) {
              setVoiceActivity('speaking');
              setTimeout(() => { if (!cancelled) setVoiceActivity('listening'); }, 2000);
            } else {
              setVoiceActivity('listening');
            }
            setIsThinking(false);
          }
        } else {
          setVoiceActivity('listening');
          setIsThinking(false);
        }
      } catch (err: any) {
        // Silently swallow rate-limit errors — the next poll will succeed
        if (!String(err.message).includes('429') && !String(err.message).toLowerCase().includes('too many')) {
          setError((prev) => prev || `Transcript sync failed: ${err.message}`);
        }
        setIsThinking(false);
      }
    };

    fetchHistory();
    const intervalId = window.setInterval(fetchHistory, 5000);
    return () => { cancelled = true; window.clearInterval(intervalId); };
  }, [callSid, callStatus, introMessage]);

  // Initialize Twilio Device
  useEffect(() => {
    let mounted = true;
    let twilioDevice: Device | null = null;

    async function initTwilioDevice() {
      if (!token) {
        setDeviceReady(false);
        setDevice(null);
        setError(language === 'ar' ? 'يرجى تسجيل الدخول' : 'Please sign in');
        return;
      }
      try {
        // Check microphone
        try {
          const testStream = await navigator.mediaDevices.getUserMedia({ audio: true });
          testStream.getTracks().forEach(track => track.stop());
        } catch (micErr: any) {
          setError(language === 'ar' 
            ? 'فشل الوصول إلى الميكروفون'
            : 'Microphone access failed');
          setDeviceReady(false);
          return;
        }

        const { token: twilioToken } = await api.getTwilioToken();
        if (!mounted) return;

        twilioDevice = new Device(twilioToken, {
          codecPreferences: [Call.Codec.Opus, Call.Codec.PCMU],
          edge: 'ashburn',
        });

        twilioDevice.on('registered', () => {
          setDeviceReady(true);
          setError('');
        });

        twilioDevice.on('error', (error: any) => {
          setError(`Device error: ${error.message}`);
          setDeviceReady(false);
        });

        twilioDevice.on('unregistered', () => setDeviceReady(false));

        await twilioDevice.register();
        if (!mounted) return;
        setDevice(twilioDevice);
      } catch (err: any) {
        setError(`Initialization failed: ${err.message}`);
        setDeviceReady(false);
      }
    }

    initTwilioDevice();
    return () => { mounted = false; if (twilioDevice) twilioDevice.destroy(); };
  }, [token, language]);

  const startCall = async () => {
    if (!device) {
      setError('Device not initialized');
      return;
    }

    try {
      setCallStatus('connecting');
      setError('');
      setTranscript([]);
      setIntroMessage(null);
      setCallSid('');

      const outgoingCall = await device.connect({ 
        params: { To: '+1234567890' },
        rtcConstraints: {
          audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true }
        }
      });
      setCall(outgoingCall);

      outgoingCall.on('accept', () => {
        setCallStatus('connected');
        setVoiceActivity('listening');
        const callParams = (outgoingCall as any)?.parameters || {};
        const sid = callParams.CallSid || callParams.callSid || '';
        if (sid) setCallSid(sid);
        
        const greeting: Message = {
          role: 'system',
          content: language === 'ar' 
            ? 'مرحباً بك. كيف يمكنني مساعدتك؟'
            : 'Welcome. How can I help you?',
          timestamp: new Date(),
        };
        setIntroMessage(greeting);
        setTranscript([greeting]);
      });

      outgoingCall.on('disconnect', () => {
        setCallStatus('disconnected');
        setVoiceActivity('idle');
        setCall(null);
        setCallSid('');
        setTimeout(() => setCallStatus('idle'), 2000);
      });

      outgoingCall.on('cancel', () => {
        setCallStatus('idle');
        setVoiceActivity('idle');
        setCall(null);
      });

      outgoingCall.on('error', (error: any) => {
        setError(`Call error: ${error.message}`);
        setCallStatus('idle');
        setCall(null);
      });
    } catch (err: any) {
      setError(`Failed to connect: ${err.message}`);
      setCallStatus('idle');
    }
  };

  const endCall = () => {
    if (call) {
      call.disconnect();
      setCall(null);
      setCallStatus('idle');
      setCallSid('');
      setIntroMessage(null);
      setVoiceActivity('idle');
      lastMessageRef.current = null;
    }
  };

  const toggleMute = () => {
    if (call) {
      call.mute(!isMuted);
      setIsMuted(!isMuted);
    }
  };

  const getStatusInfo = () => {
    const statusMap = {
      idle: { text: language === 'ar' ? 'جاهز' : 'Ready', color: 'bg-gray-400' },
      connecting: { text: language === 'ar' ? 'جاري الاتصال...' : 'Connecting...', color: 'bg-yellow-500' },
      connected: { text: language === 'ar' ? 'متصل' : 'Connected', color: 'bg-green-500' },
      disconnected: { text: language === 'ar' ? 'منتهي' : 'Ended', color: 'bg-gray-500' }
    };
    return statusMap[callStatus];
  };

  return (
    <div className="max-w-5xl mx-auto">
      {/* Header */}
      <div className="mb-6 flex items-center gap-4">
        <div className={clsx(
          'w-12 h-12 rounded-2xl flex items-center justify-center text-white text-xl font-bold shadow-lg flex-shrink-0',
          callStatus === 'connected' ? 'bg-emerald-600 shadow-emerald-500/30' : 'bg-gray-500'
        )}>ل</div>
        <div>
          <h1 className="text-2xl lg:text-3xl font-bold">
            {language === 'ar' ? 'ليان — مساعدة بيميدكس' : 'Leyan — Bimedx Assistant'}
          </h1>
          <p className={clsx('text-sm mt-0.5', theme === 'dark' ? 'text-gray-400' : 'text-gray-500')}>
            {language === 'ar' ? 'المساعدة الصوتية الذكية لحجز المواعيد' : 'AI voice assistant · appointment booking'}
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-5">

        {/* ── CALL PANEL (3/5) ── */}
        <div className={clsx(
          'lg:col-span-3 rounded-2xl border flex flex-col overflow-hidden',
          theme === 'dark' ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'
        )}>
          {/* Status bar */}
          <div className={clsx(
            'flex items-center justify-between px-5 py-3 border-b',
            theme === 'dark' ? 'border-gray-700' : 'border-gray-200'
          )}>
            <div className="flex items-center gap-2">
              <div className={clsx(
                'w-2.5 h-2.5 rounded-full',
                callStatus === 'connected'    ? 'bg-green-400 animate-pulse' :
                callStatus === 'connecting'   ? 'bg-yellow-400 animate-pulse' :
                callStatus === 'disconnected' ? 'bg-gray-400' : 'bg-gray-400'
              )} />
              <span className="font-semibold text-sm">
                {callStatus === 'connected'
                  ? (language === 'ar' ? 'متصل' : 'Connected')
                  : callStatus === 'connecting'
                    ? (language === 'ar' ? 'جاري الاتصال...' : 'Connecting...')
                    : callStatus === 'disconnected'
                      ? (language === 'ar' ? 'انتهت المكالمة' : 'Call Ended')
                      : (language === 'ar' ? 'جاهز' : 'Ready')}
              </span>
            </div>
            {callStatus === 'connected' && (
              <span className={clsx('text-xs font-mono tabular-nums', theme === 'dark' ? 'text-gray-400' : 'text-gray-500')}>
                {String(Math.floor(callSeconds / 60)).padStart(2, '0')}:
                {String(callSeconds % 60).padStart(2, '0')}
              </span>
            )}
          </div>

          {/* Conversation area */}
          <div className={clsx(
            'flex-1 overflow-y-auto px-4 py-4 space-y-3 h-[400px] lg:h-[450px]',
            theme === 'dark' ? 'bg-gray-900/40' : 'bg-gray-50'
          )}>
            {/* Idle placeholder */}
            {callStatus === 'idle' && demoConvo.length === 0 && (
              <div className="flex flex-col items-center justify-center h-full gap-4 text-center">
                <div className="w-20 h-20 rounded-full bg-emerald-500/10 flex items-center justify-center">
                  <span className="text-3xl font-bold text-emerald-600">ل</span>
                </div>
                <div>
                  <p className="font-semibold text-base">
                    {language === 'ar' ? 'ليان — مستشفى بيميدكس' : 'Leyan — Bimedx Hospital'}
                  </p>
                  <p className={clsx('text-sm mt-1', theme === 'dark' ? 'text-gray-400' : 'text-gray-500')}>
                    {language === 'ar' ? 'اضغط "اتصل" للبدء' : 'Press "Call" to start'}
                  </p>
                </div>
              </div>
            )}

            {/* Connecting animation */}
            {callStatus === 'connecting' && (
              <div className="flex flex-col items-center justify-center h-full gap-4">
                <div className="relative w-20 h-20">
                  <div className="absolute inset-0 w-20 h-20 rounded-full bg-green-500/20 animate-ping" />
                  <div className="w-20 h-20 rounded-full bg-green-500/10 flex items-center justify-center relative">
                    <IconPhone size={30} className="text-green-500 animate-pulse" />
                  </div>
                </div>
                <p className="text-sm font-medium text-green-500">
                  {language === 'ar' ? 'جاري الاتصال بليان...' : 'Calling Leyan...'}
                </p>
              </div>
            )}

            {/* Conversation bubbles */}
            {demoConvo.map((msg, idx) => (
              <div key={idx} className={clsx('flex', msg.role === 'patient' ? 'justify-end' : 'justify-start')}>
                {msg.role === 'va' && (
                  <div className="w-7 h-7 rounded-full bg-emerald-600 flex items-center justify-center text-white text-xs font-bold mr-2 mt-1 flex-shrink-0">ل</div>
                )}
                <div className={clsx(
                  'max-w-[78%] rounded-2xl px-4 py-2.5',
                  msg.role === 'va' ? 'bg-emerald-600 text-white rounded-tl-sm' : 'bg-blue-600 text-white rounded-tr-sm'
                )}>
                  <p className="text-sm leading-relaxed" dir="rtl">{msg.text}</p>
                </div>
                {msg.role === 'patient' && (
                  <div className="w-7 h-7 rounded-full bg-blue-600 flex items-center justify-center text-white text-xs font-bold ml-2 mt-1 flex-shrink-0">أ</div>
                )}
              </div>
            ))}

            {/* Live typewriting VA bubble */}
            {demoTypeText && (
              <div className="flex justify-start">
                <div className="w-7 h-7 rounded-full bg-emerald-600 flex items-center justify-center text-white text-xs font-bold mr-2 mt-1 flex-shrink-0">ل</div>
                <div className="max-w-[78%] rounded-2xl rounded-tl-sm px-4 py-2.5 bg-emerald-600 text-white">
                  <p className="text-sm leading-relaxed" dir="rtl">
                    {demoTypeText}
                    <span className="inline-block w-0.5 h-4 bg-white opacity-80 animate-pulse ml-0.5 align-middle" />
                  </p>
                </div>
              </div>
            )}

            {/* Voice activity bars */}
            {callStatus === 'connected' && !demoTypeText && (
              <>
                {demoPhase === 'va' && (
                  <div className="flex justify-start">
                    <div className={clsx('w-7 h-7 rounded-full flex items-center justify-center mr-2 mt-1 flex-shrink-0', theme === 'dark' ? 'bg-emerald-900/40' : 'bg-emerald-100')}>
                      <IconVolume size={13} className="text-emerald-500" />
                    </div>
                    <div className={clsx('rounded-2xl rounded-tl-sm px-4 py-3', theme === 'dark' ? 'bg-gray-700' : 'bg-gray-200')}>
                      <div className="flex items-center gap-0.5">
                        {[...Array(5)].map((_, i) => (
                          <div key={i} className="w-1.5 bg-emerald-500 rounded-full animate-pulse"
                            style={{ height: `${10 + (i % 3) * 5}px`, animationDelay: `${i * 0.12}s` }} />
                        ))}
                      </div>
                    </div>
                  </div>
                )}
                {demoPhase === 'listening' && (
                  <div className="flex justify-end">
                    <div className={clsx('rounded-2xl rounded-tr-sm px-4 py-3', theme === 'dark' ? 'bg-gray-700' : 'bg-gray-200')}>
                      <div className="flex items-center gap-1">
                        <div className="flex items-center gap-0.5">
                          {[...Array(5)].map((_, i) => (
                            <div key={i} className="w-1.5 bg-blue-500 rounded-full animate-pulse"
                              style={{ height: `${8 + (i % 4) * 4}px`, animationDelay: `${i * 0.09}s` }} />
                          ))}
                        </div>
                        <IconMicrophone size={13} className="text-blue-400 ml-1" />
                      </div>
                    </div>
                    <div className="w-7 h-7 rounded-full bg-blue-600 flex items-center justify-center text-white text-xs font-bold ml-2 mt-1 flex-shrink-0">أ</div>
                  </div>
                )}
              </>
            )}

            {demoPhase === 'done' && (
              <div className="flex justify-center py-2">
                <div className="px-4 py-2 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-500 text-sm font-medium">
                  ✓ {language === 'ar' ? 'تم تأكيد الحجز' : 'Booking Confirmed'}
                </div>
              </div>
            )}

            <div ref={demoEndRef} />
          </div>

          {/* Call controls */}
          <div className={clsx('p-4 border-t', theme === 'dark' ? 'border-gray-700' : 'border-gray-200')}>
            {(callStatus === 'idle' || callStatus === 'disconnected') && (
              <button
                onClick={handleCallButton}
                className="w-full flex items-center justify-center gap-2 py-4 rounded-xl font-bold text-base bg-green-600 hover:bg-green-700 active:scale-95 text-white transition-all shadow-lg shadow-green-500/25"
              >
                <IconPhone size={22} />
                {callStatus === 'disconnected'
                  ? (language === 'ar' ? '↺ اتصل مجدداً' : '↺ Call Again')
                  : (language === 'ar' ? 'اتصل بليان' : 'Call Leyan')}
              </button>
            )}
            {callStatus === 'connecting' && (
              <button disabled className="w-full flex items-center justify-center gap-2 py-4 rounded-xl font-bold text-base bg-yellow-500/20 text-yellow-500 cursor-not-allowed">
                <IconLoader2 size={20} className="animate-spin" />
                {language === 'ar' ? 'جاري الاتصال...' : 'Connecting...'}
              </button>
            )}
            {callStatus === 'connected' && (
              <div className="flex gap-3">
                <button
                  onClick={() => setIsMuted(!isMuted)}
                  className={clsx(
                    'flex items-center justify-center gap-2 py-3 px-5 rounded-xl font-medium transition-colors',
                    isMuted ? 'bg-yellow-500 hover:bg-yellow-600 text-white' : 'bg-blue-600 hover:bg-blue-700 text-white'
                  )}
                >
                  {isMuted ? <IconMicrophoneOff size={18} /> : <IconMicrophone size={18} />}
                  {isMuted ? (language === 'ar' ? 'إلغاء الكتم' : 'Unmute') : (language === 'ar' ? 'كتم' : 'Mute')}
                </button>
                <button
                  onClick={handleEndCall}
                  className="flex-1 flex items-center justify-center gap-2 py-3 rounded-xl font-medium bg-red-600 hover:bg-red-700 text-white transition-colors"
                >
                  <IconPhoneOff size={20} />
                  {language === 'ar' ? 'إنهاء المكالمة' : 'End Call'}
                </button>
              </div>
            )}
          </div>
        </div>

        {/* ── BOOKING FORM PANEL (2/5) ── */}
        <div className={clsx(
          'lg:col-span-2 rounded-2xl border flex flex-col',
          theme === 'dark' ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'
        )}>
          <div className={clsx('px-4 py-3 border-b flex items-center gap-2', theme === 'dark' ? 'border-gray-700' : 'border-gray-200')}>
            <span className="text-base">📋</span>
            <span className="font-semibold text-sm">
              {language === 'ar' ? 'نموذج الحجز' : 'Booking Details'}
            </span>
            {demoPhase === 'done' && (
              <span className="ml-auto text-xs bg-emerald-500/10 text-emerald-500 px-2 py-0.5 rounded-full font-medium">
                ✓ {language === 'ar' ? 'مؤكد' : 'Confirmed'}
              </span>
            )}
          </div>

          <div className="flex-1 px-4 py-3 space-y-2">
            {([
              {
                label: language === 'ar' ? 'التاريخ' : 'Date',
                value: new Date().toLocaleDateString(language === 'ar' ? 'ar-SA' : 'en-US', { day: '2-digit', month: '2-digit', year: 'numeric' }),
                filled: callStatus !== 'idle',
              },
              {
                label: language === 'ar' ? 'التخصص' : 'Specialty',
                value: language === 'ar' ? 'الأمراض الجلدية' : 'Dermatology',
                filled: demoStep >= 1,
              },
              {
                label: language === 'ar' ? 'اسم المريض' : 'Patient Name',
                value: 'منصور محمد منصور',
                filled: demoStep >= 2,
              },
              {
                label: language === 'ar' ? 'رقم الهاتف' : 'Phone',
                value: '٠١٠٩٥٠١٣٥٣٦',
                filled: demoStep >= 3,
              },
              {
                label: language === 'ar' ? 'تاريخ الميلاد' : 'Date of Birth',
                value: '26/08/2001',
                filled: demoStep >= 4,
              },
              {
                label: language === 'ar' ? 'اليوم' : 'Day',
                value: language === 'ar' ? 'الثلاثاء' : 'Tuesday',
                filled: demoStep >= 5,
              },
              {
                label: language === 'ar' ? 'الوقت' : 'Time',
                value: language === 'ar' ? '٣:٠٠ مساءً' : '3:00 PM',
                filled: demoStep >= 6,
              },
            ] as { label: string; value: string; filled: boolean }[]).map(({ label, value, filled }) => (
              <div key={label} className={clsx(
                'flex items-center justify-between py-2 px-3 rounded-lg transition-all duration-500',
                filled
                  ? theme === 'dark' ? 'bg-emerald-900/20 border border-emerald-800/40' : 'bg-emerald-50 border border-emerald-200/60'
                  : theme === 'dark' ? 'bg-gray-700/30 border border-transparent' : 'bg-gray-50 border border-transparent'
              )}>
                <span className={clsx('text-xs font-medium', theme === 'dark' ? 'text-gray-400' : 'text-gray-500')}>
                  {label}
                </span>
                {filled
                  ? <span className="text-xs font-semibold text-emerald-500 flex items-center gap-1" dir="rtl">
                      <span className="text-emerald-400 text-xs">✓</span> {value}
                    </span>
                  : <span className={clsx('text-xs', theme === 'dark' ? 'text-gray-600' : 'text-gray-300')}>—</span>
                }
              </div>
            ))}
          </div>

          {/* Presenter cue — shows next patient line during listening phase */}
          {callStatus === 'connected' && demoPhase === 'listening'
            && demoStep >= 0 && demoStep < DEMO_SCRIPT.length
            && (DEMO_SCRIPT[demoStep] as any).patient && (
            <div className={clsx(
              'px-4 py-3 border-t text-right',
              theme === 'dark' ? 'border-gray-700 bg-gray-900/30' : 'border-gray-100 bg-gray-50'
            )}>
              <p className={clsx('text-xs mb-1', theme === 'dark' ? 'text-gray-500' : 'text-gray-400')} dir="rtl">
                💬 {language === 'ar' ? 'قل الآن:' : 'Say now:'}
              </p>
              <p className="text-xs text-emerald-500 font-medium" dir="rtl">
                {(DEMO_SCRIPT[demoStep] as any).patient}
              </p>
            </div>
          )}
        </div>

      </div>
    </div>
  );
};

export default VoiceAgentClean;