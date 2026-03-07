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

  // ── Scripted Presenter Demo Mode ─────────────────────────────────────────
  // VA asks hardcoded questions → TTS speaks them → pause for presenter → repeat
  const [demoMode, setDemoMode] = useState(false);
  // -1 = not started, 0..N = current step index
  const [demoStep, setDemoStep] = useState(-1);
  // 'idle' | 'thinking' | 'va' | 'listening' | 'done'
  const [demoPhase, setDemoPhase] = useState<'idle' | 'thinking' | 'va' | 'listening' | 'done'>('idle');
  // Messages shown in left conversation panel
  const [demoConvo, setDemoConvo] = useState<{ role: 'va' | 'patient'; text: string }[]>([]);
  // Text currently being typed into the active VA bubble
  const [demoTypeText, setDemoTypeText] = useState('');
  // 0→100 countdown bar for the listening phase
  const [demoListenPct, setDemoListenPct] = useState(100);
  const [demoAudioPlaying, setDemoAudioPlaying] = useState(false);
  const demoAbortRef = useRef(false);
  const demoEndRef = useRef<HTMLDivElement>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);

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

  // G.711 μ-law → 16-bit PCM decoder (no external lib)
  function mulawDecode(byte: number): number {
    byte = ~byte & 0xff;
    const sign = byte & 0x80;
    const exp  = (byte >> 4) & 0x07;
    const mant = byte & 0x0f;
    let sample = ((mant << 3) + 33) << exp;
    sample -= 33;
    return sign ? -sample : sample;
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
      setDemoAudioPlaying(true);
      src.start();
      await new Promise<void>(r => { src.onended = () => r(); });
    } catch { /* non-fatal */ }
    finally { setDemoAudioPlaying(false); }
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

  // ── Fully hardcoded scripted demo — no LLM calls ──────────────────────
  const runScriptedDemo = useCallback(async () => {
    demoAbortRef.current = false;
    setDemoConvo([]);
    setDemoTypeText('');
    setDemoStep(-1);
    setDemoPhase('idle');
    setDemoListenPct(100);

    for (let i = 0; i < DEMO_SCRIPT.length; i++) {
      if (demoAbortRef.current) break;
      const step = DEMO_SCRIPT[i];
      setDemoStep(i);

      // 1. Typewrite VA line while TTS synthesises in background
      setDemoPhase('va');
      setDemoTypeText('');
      const ttsPromise = api.synthesizeSpeech(step.va, 'saudi-tts').catch(() => null);
      await typewrite(step.va, demoAbortRef);
      if (demoAbortRef.current) break;

      // Commit VA bubble to conversation
      setDemoConvo(prev => [...prev, { role: 'va', text: step.va }]);
      setDemoTypeText('');

      // 2. Play Ahmed Saudi TTS
      try {
        const ttsRes = await ttsPromise;
        if (ttsRes && (ttsRes as any).audio) {
          await playMulawAudio((ttsRes as any).audio, (ttsRes as any).sampleRate || 8000);
        }
      } catch { /* non-fatal */ }
      if (demoAbortRef.current) break;

      // 3. If there is a patient cue, run the listening countdown then show it
      if (step.patient) {
        setDemoPhase('listening');
        setDemoListenPct(100);
        const ticks = 60;
        const interval = LISTEN_MS / ticks;
        for (let t = ticks; t >= 0; t--) {
          if (demoAbortRef.current) break;
          setDemoListenPct((t / ticks) * 100);
          await new Promise(r => setTimeout(r, interval));
        }
        if (demoAbortRef.current) break;

        // Show patient bubble
        setDemoConvo(prev => [...prev, { role: 'patient', text: step.patient as string }]);
        await new Promise(r => setTimeout(r, 500));
      }
      if (demoAbortRef.current) break;
    }

    setDemoPhase('done');
    setDemoTypeText('');
  }, [playMulawAudio, typewrite]);

  const stopDemo = () => {
    demoAbortRef.current = true;
    setDemoPhase('idle');
    setDemoTypeText('');
    setDemoStep(-1);
    setDemoListenPct(100);
  };

  const resetDemo = () => {
    stopDemo();
    setDemoConvo([]);
  };

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

  const statusInfo = getStatusInfo();

  return (
    <div className="max-w-4xl mx-auto">
      {/* Header */}
      <div className="mb-6 flex items-center justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-2xl lg:text-3xl font-bold">
            {language === 'ar' ? 'المساعد الصوتي' : 'Voice Agent'}
          </h1>
          <p className={clsx('mt-1', theme === 'dark' ? 'text-gray-400' : 'text-gray-500')}>
            {language === 'ar' ? 'تحدث مع المساعد الطبي الذكي' : 'Talk to the AI medical assistant'}
          </p>
        </div>
        {/* Mode Toggle */}
        <div className={clsx(
          'flex rounded-xl overflow-hidden border text-sm font-medium',
          theme === 'dark' ? 'border-gray-700' : 'border-gray-200'
        )}>
          <button
            onClick={() => setDemoMode(false)}
            className={clsx(
              'px-4 py-2 flex items-center gap-2 transition-colors',
              !demoMode
                ? 'bg-blue-600 text-white'
                : theme === 'dark' ? 'bg-gray-800 text-gray-300 hover:bg-gray-700' : 'bg-white text-gray-600 hover:bg-gray-50'
            )}
          >
            <IconPhone size={15} />
            {language === 'ar' ? 'مكالمة' : 'Call'}
          </button>
          <button
            onClick={() => setDemoMode(true)}
            className={clsx(
              'px-4 py-2 flex items-center gap-2 transition-colors',
              demoMode
                ? 'bg-emerald-600 text-white'
                : theme === 'dark' ? 'bg-gray-800 text-gray-300 hover:bg-gray-700' : 'bg-white text-gray-600 hover:bg-gray-50'
            )}
          >
            <IconPlayerPlay size={15} />
            {language === 'ar' ? 'عرض تجريبي' : 'Live Demo'}
          </button>
        </div>
      </div>

      {/* ── SCRIPTED PRESENTER DEMO MODE ─────────────────────────────────── */}
      {demoMode ? (
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-5">

          {/* ── LEFT: Live conversation (3/5) ── */}
          <div className={clsx(
            'lg:col-span-3 rounded-2xl border flex flex-col overflow-hidden',
            theme === 'dark' ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'
          )}>
            {/* Header */}
            <div className={clsx(
              'flex items-center justify-between px-5 py-3 border-b',
              theme === 'dark' ? 'border-gray-700' : 'border-gray-200'
            )}>
              <div className="flex items-center gap-2">
                <div className={clsx(
                  'w-2.5 h-2.5 rounded-full',
                  demoPhase === 'va' ? 'bg-emerald-400 animate-pulse' :
                  demoPhase === 'listening' ? 'bg-green-400 animate-pulse' :
                  demoPhase === 'done' ? 'bg-blue-400' : 'bg-gray-400'
                )} />
                <span className="font-semibold text-sm">
                  {language === 'ar' ? 'ليان — المساعدة الصوتية' : 'Leyan — Voice Agent'}
                </span>
                {demoAudioPlaying && (
                  <span className="flex items-center gap-1 text-xs text-emerald-400 animate-pulse">
                    <IconVolume size={13} /> {language === 'ar' ? 'تتحدث...' : 'Speaking...'}
                  </span>
                )}
              </div>
              <div className="flex items-center gap-2">
                {demoStep >= 0 && demoPhase !== 'idle' && demoPhase !== 'done' && (
                  <span className={clsx('text-xs', theme === 'dark' ? 'text-gray-400' : 'text-gray-500')}>
                    {demoStep + 1} / {DEMO_SCRIPT.length}
                  </span>
                )}
                {demoPhase === 'done' && (
                  <span className="text-xs text-emerald-500 font-medium">
                    {language === 'ar' ? 'اكتمل الحجز ✓' : 'Booking Done ✓'}
                  </span>
                )}
                <button
                  onClick={resetDemo}
                  className={clsx(
                    'text-xs px-3 py-1 rounded-lg transition-colors',
                    theme === 'dark' ? 'text-gray-400 hover:bg-gray-700' : 'text-gray-500 hover:bg-gray-100'
                  )}
                >
                  {language === 'ar' ? 'إعادة' : 'Reset'}
                </button>
              </div>
            </div>

            {/* Conversation area */}
            <div className={clsx(
              'flex-1 overflow-y-auto px-4 py-4 space-y-3',
              'h-[400px] lg:h-[450px]',
              theme === 'dark' ? 'bg-gray-900/40' : 'bg-gray-50'
            )}>
              {demoConvo.length === 0 && demoPhase === 'idle' && (
                <div className="flex flex-col items-center justify-center h-full gap-4 text-center">
                  <div className="w-16 h-16 rounded-full bg-emerald-500/10 flex items-center justify-center">
                    <IconPlayerPlay size={30} className="text-emerald-500" />
                  </div>
                  <div>
                    <p className="font-semibold text-base">
                      {language === 'ar' ? 'عرض حجز موعد تجريبي' : 'Appointment Booking Demo'}
                    </p>
                    <p className={clsx('text-sm mt-1', theme === 'dark' ? 'text-gray-400' : 'text-gray-500')}>
                      {language === 'ar'
                        ? 'ليان ستسأل — أنت تجيب — الصوت سعودي'
                        : 'Leyan asks · you answer · Ahmed Saudi voice'}
                    </p>
                  </div>
                </div>
              )}

              {/* Committed conversation bubbles */}
              {demoConvo.map((msg, idx) => (
                <div
                  key={idx}
                  className={clsx(
                    'flex',
                    msg.role === 'patient' ? 'justify-end' : 'justify-start'
                  )}
                >
                  {msg.role === 'va' && (
                    <div className="w-7 h-7 rounded-full bg-emerald-600 flex items-center justify-center text-white text-xs font-bold mr-2 mt-1 flex-shrink-0">
                      ل
                    </div>
                  )}
                  <div className={clsx(
                    'max-w-[78%] rounded-2xl px-4 py-2.5',
                    msg.role === 'va'
                      ? 'bg-emerald-600 text-white rounded-tl-sm'
                      : 'bg-blue-600 text-white rounded-tr-sm'
                  )}>
                    <p className="text-sm leading-relaxed" dir="rtl">{msg.text}</p>
                  </div>
                  {msg.role === 'patient' && (
                    <div className="w-7 h-7 rounded-full bg-blue-600 flex items-center justify-center text-white text-xs font-bold ml-2 mt-1 flex-shrink-0">
                      م
                    </div>
                  )}
                </div>
              ))}

              {/* Live typewriting VA bubble */}
              {demoTypeText && (
                <div className="flex justify-start">
                  <div className="w-7 h-7 rounded-full bg-emerald-600 flex items-center justify-center text-white text-xs font-bold mr-2 mt-1 flex-shrink-0">
                    ل
                  </div>
                  <div className="max-w-[78%] rounded-2xl rounded-tl-sm px-4 py-2.5 bg-emerald-600 text-white">
                    <p className="text-sm leading-relaxed" dir="rtl">
                      {demoTypeText}
                      <span className="inline-block w-0.5 h-4 bg-white opacity-80 animate-pulse ml-0.5 align-middle" />
                    </p>
                  </div>
                </div>
              )}

              {/* Listening indicator */}
              {demoPhase === 'listening' && (
                <div className="flex justify-start">
                  <div className="w-7 h-7 rounded-full bg-green-600 flex items-center justify-center text-white mr-2 flex-shrink-0">
                    <IconEar size={14} />
                  </div>
                  <div className={clsx(
                    'rounded-2xl rounded-tl-sm px-4 py-3 w-48',
                    theme === 'dark' ? 'bg-gray-700' : 'bg-gray-200'
                  )}>
                    <p className={clsx('text-xs mb-2 font-medium', theme === 'dark' ? 'text-green-400' : 'text-green-600')} dir="rtl">
                      {language === 'ar' ? 'يستمع إليك...' : 'Listening...'}
                    </p>
                    <div className={clsx('h-1.5 w-full rounded-full overflow-hidden', theme === 'dark' ? 'bg-gray-600' : 'bg-gray-300')}>
                      <div
                        className="h-full bg-green-500 rounded-full transition-all duration-100"
                        style={{ width: `${demoListenPct}%` }}
                      />
                    </div>
                  </div>
                </div>
              )}

              <div ref={demoEndRef} />
            </div>

            {/* Run / Stop button */}
            <div className={clsx('p-4 border-t', theme === 'dark' ? 'border-gray-700' : 'border-gray-200')}>
              {demoPhase === 'idle' || demoPhase === 'done' ? (
                <button
                  onClick={runScriptedDemo}
                  className="w-full flex items-center justify-center gap-2 py-3 rounded-xl font-bold text-sm bg-emerald-600 hover:bg-emerald-700 active:scale-95 text-white transition-all"
                >
                  <IconPlayerPlay size={18} />
                  {demoPhase === 'done'
                    ? (language === 'ar' ? '↺ إعادة العرض' : '↺ Replay Demo')
                    : (language === 'ar' ? '▶ تشغيل العرض التجريبي' : '▶ Start Demo')}
                </button>
              ) : (
                <button
                  onClick={stopDemo}
                  className="w-full flex items-center justify-center gap-2 py-3 rounded-xl font-bold text-sm bg-red-600 hover:bg-red-700 active:scale-95 text-white transition-all"
                >
                  ■ {language === 'ar' ? 'إيقاف العرض' : 'Stop Demo'}
                </button>
              )}
            </div>
          </div>

          {/* ── RIGHT: Live script panel (2/5) ── */}
          <div className={clsx(
            'lg:col-span-2 rounded-2xl border flex flex-col overflow-hidden',
            theme === 'dark' ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'
          )}>
            <div className={clsx('px-4 py-3 border-b flex items-center gap-2', theme === 'dark' ? 'border-gray-700' : 'border-gray-200')}>
              <IconMessage size={15} className="text-emerald-500" />
              <span className="font-semibold text-sm">
                {language === 'ar' ? 'النص الحي' : 'Live Script'}
              </span>
            </div>
            <div className="flex-1 overflow-y-auto px-3 py-3 space-y-2">
              {DEMO_SCRIPT.map((step, idx) => {
                const isActive  = demoStep === idx && (demoPhase === 'va' || demoPhase === 'listening');
                const isDone    = demoStep > idx || demoPhase === 'done';

                return (
                  <div
                    key={idx}
                    className={clsx(
                      'rounded-xl px-3 py-2.5 border transition-all duration-300',
                      isActive
                        ? 'border-emerald-500 bg-emerald-500/10 shadow-sm shadow-emerald-500/20'
                        : isDone
                          ? theme === 'dark' ? 'border-gray-700 bg-gray-900/30 opacity-60' : 'border-gray-100 bg-gray-50 opacity-60'
                          : theme === 'dark' ? 'border-gray-700/50 opacity-30' : 'border-gray-100 opacity-30'
                    )}
                  >
                    {/* Step number + live status */}
                    <div className="flex items-center gap-1.5 mb-1.5">
                      <span className={clsx(
                        'text-xs font-bold w-5 h-5 rounded-full flex items-center justify-center flex-shrink-0',
                        isActive ? 'bg-emerald-500 text-white animate-pulse' :
                        isDone   ? 'bg-emerald-600/30 text-emerald-500' :
                                   theme === 'dark' ? 'bg-gray-700 text-gray-500' : 'bg-gray-200 text-gray-400'
                      )}>
                        {isDone ? '✓' : idx + 1}
                      </span>
                      {isActive && (
                        <span className="text-xs text-emerald-400 animate-pulse font-medium">
                          {demoPhase === 'va'
                            ? (language === 'ar' ? 'يتحدث...' : 'Speaking...')
                            : (language === 'ar' ? 'يستمع...' : 'Listening...')}
                        </span>
                      )}
                    </div>

                    {/* VA line — shows typewriter cursor when active */}
                    <div className={clsx(
                      'text-xs px-2 py-1.5 rounded-lg mb-1 text-right',
                      isDone || isActive
                        ? 'bg-emerald-500/10 text-emerald-400'
                        : theme === 'dark' ? 'bg-gray-700/40 text-gray-600' : 'bg-gray-100 text-gray-400'
                    )} dir="rtl">
                      {isActive && demoPhase === 'va' && demoTypeText
                        ? <>{demoTypeText}<span className="inline-block w-0.5 h-3 bg-emerald-400 animate-pulse ml-0.5 align-middle" /></>
                        : <>🤖 {step.va}</>}
                    </div>

                    {/* Patient cue — always shown if it exists */}
                    {step.patient && (
                      <div className={clsx(
                        'text-xs px-2 py-1.5 rounded-lg text-right',
                        isDone || isActive
                          ? 'bg-blue-500/10 text-blue-400'
                          : theme === 'dark' ? 'bg-gray-700/40 text-gray-600' : 'bg-gray-100 text-gray-400'
                      )} dir="rtl">
                        👤 {step.patient}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

        </div>
      ) : (
      /* ── PHONE CALL MODE ────────────────────────────────────────────────── */
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main Call Panel */}
        <div className={clsx(
          'lg:col-span-2 rounded-xl border p-6',
          theme === 'dark' ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'
        )}>
          {/* Status Bar */}
          <div className="flex items-center justify-between mb-6 pb-4 border-b border-gray-200 dark:border-gray-700">
            <div className="flex items-center gap-3">
              <div className={clsx('w-3 h-3 rounded-full', statusInfo.color, 
                callStatus === 'connecting' && 'animate-pulse')} />
              <span className="font-medium">{statusInfo.text}</span>
              {callSid && (
                <span className={clsx('text-xs', theme === 'dark' ? 'text-gray-500' : 'text-gray-400')}>
                  #{callSid.slice(-8)}
                </span>
              )}
            </div>
            {deviceReady && (
              <span className="flex items-center gap-1 text-green-500 text-sm">
                <IconCheck size={16} />
                {language === 'ar' ? 'جاهز' : 'Ready'}
              </span>
            )}
          </div>

          {/* Error Display */}
          {error && (
            <div className="mb-4 p-3 rounded-lg bg-red-500/10 border border-red-500/20 flex items-center gap-2 text-red-500">
              <IconAlertCircle size={18} />
              <span className="text-sm">{error}</span>
            </div>
          )}

          {/* Voice Activity Indicator */}
          {callStatus === 'connected' && (
            <div className={clsx(
              'mb-6 p-4 rounded-xl flex items-center gap-4',
              theme === 'dark' ? 'bg-gray-700/50' : 'bg-gray-50'
            )}>
              {voiceActivity === 'listening' && (
                <>
                  <div className="p-3 rounded-full bg-green-500">
                    <IconEar size={24} className="text-white" />
                  </div>
                  <div>
                    <p className="font-medium text-green-500">
                      {language === 'ar' ? 'أستمع إليك...' : 'Listening...'}
                    </p>
                    <p className={clsx('text-sm', theme === 'dark' ? 'text-gray-400' : 'text-gray-500')}>
                      {language === 'ar' ? 'تحدث الآن' : 'Speak now'}
                    </p>
                  </div>
                  <div className="flex items-center gap-1 ml-auto">
                    {[...Array(4)].map((_, i) => (
                      <div key={i} className="w-1 bg-green-500 rounded-full animate-pulse"
                        style={{ height: `${12 + i * 4}px`, animationDelay: `${i * 0.15}s` }} />
                    ))}
                  </div>
                </>
              )}
              {voiceActivity === 'processing' && (
                <>
                  <div className="p-3 rounded-full bg-yellow-500">
                    <IconLoader2 size={24} className="text-white animate-spin" />
                  </div>
                  <div>
                    <p className="font-medium text-yellow-500">
                      {language === 'ar' ? 'جاري المعالجة...' : 'Processing...'}
                    </p>
                    <p className={clsx('text-sm', theme === 'dark' ? 'text-gray-400' : 'text-gray-500')}>
                      {language === 'ar' ? 'أفكر في إجابتك' : 'Thinking...'}
                    </p>
                  </div>
                </>
              )}
              {voiceActivity === 'speaking' && (
                <>
                  <div className="p-3 rounded-full bg-blue-500">
                    <IconVolume size={24} className="text-white" />
                  </div>
                  <div>
                    <p className="font-medium text-blue-500">
                      {language === 'ar' ? 'أتحدث...' : 'Speaking...'}
                    </p>
                    <p className={clsx('text-sm', theme === 'dark' ? 'text-gray-400' : 'text-gray-500')}>
                      {language === 'ar' ? 'استمع للإجابة' : 'Listen to response'}
                    </p>
                  </div>
                </>
              )}
            </div>
          )}

          {/* Call Controls */}
          <div className="flex flex-wrap gap-3">
            {callStatus === 'idle' && (
              <button
                onClick={startCall}
                disabled={!deviceReady}
                className={clsx(
                  'flex-1 flex items-center justify-center gap-2 py-4 px-6 rounded-xl font-medium transition-colors',
                  deviceReady 
                    ? 'bg-green-600 hover:bg-green-700 text-white'
                    : 'bg-gray-300 dark:bg-gray-700 cursor-not-allowed text-gray-500'
                )}
              >
                <IconPhone size={20} />
                {language === 'ar' ? 'ابدأ المكالمة' : 'Start Call'}
              </button>
            )}

            {(callStatus === 'connected' || callStatus === 'connecting') && (
              <>
                <button
                  onClick={toggleMute}
                  className={clsx(
                    'flex items-center justify-center gap-2 py-3 px-5 rounded-xl font-medium transition-colors',
                    isMuted 
                      ? 'bg-yellow-500 hover:bg-yellow-600 text-white'
                      : 'bg-blue-600 hover:bg-blue-700 text-white'
                  )}
                >
                  {isMuted ? <IconMicrophoneOff size={20} /> : <IconMicrophone size={20} />}
                  {isMuted ? (language === 'ar' ? 'إلغاء الكتم' : 'Unmute') : (language === 'ar' ? 'كتم' : 'Mute')}
                </button>
                <button
                  onClick={endCall}
                  className="flex-1 flex items-center justify-center gap-2 py-3 px-5 rounded-xl font-medium bg-red-600 hover:bg-red-700 text-white transition-colors"
                >
                  <IconPhoneOff size={20} />
                  {language === 'ar' ? 'إنهاء' : 'End Call'}
                </button>
              </>
            )}

            <button
              onClick={() => setShowSettings(!showSettings)}
              className={clsx(
                'p-3 rounded-xl transition-colors',
                theme === 'dark' ? 'hover:bg-gray-700' : 'hover:bg-gray-100',
                showSettings && (theme === 'dark' ? 'bg-gray-700' : 'bg-gray-100')
              )}
            >
              <IconSettings size={20} />
            </button>
          </div>

          {/* Voice Settings */}
          {showSettings && (
            <div className={clsx(
              'mt-4 p-4 rounded-xl',
              theme === 'dark' ? 'bg-gray-700/50' : 'bg-gray-50'
            )}>
              <h3 className="font-medium mb-3">
                {language === 'ar' ? 'إعدادات الصوت' : 'Voice Settings'}
              </h3>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className={clsx(
                    'block text-sm mb-1',
                    theme === 'dark' ? 'text-gray-400' : 'text-gray-500'
                  )}>
                    {language === 'ar' ? 'اللهجة' : 'Dialect'}
                  </label>
                  <select
                    value={dialectPreference}
                    onChange={(e) => setDialectPreference(e.target.value as 'auto' | 'egypt' | 'saudi')}
                    className={clsx(
                      'w-full px-3 py-2 rounded-lg border transition-colors',
                      theme === 'dark'
                        ? 'bg-gray-800 border-gray-600 text-white'
                        : 'bg-white border-gray-200 text-gray-900'
                    )}
                  >
                    <option value="auto">{language === 'ar' ? 'تلقائي' : 'Auto'}</option>
                    <option value="egypt">{language === 'ar' ? 'مصري' : 'Egyptian'}</option>
                    <option value="saudi">{language === 'ar' ? 'سعودي' : 'Saudi'}</option>
                  </select>
                </div>
                <div>
                  <label className={clsx(
                    'block text-sm mb-1',
                    theme === 'dark' ? 'text-gray-400' : 'text-gray-500'
                  )}>
                    {language === 'ar' ? 'الصوت' : 'Voice'}
                  </label>
                  <select
                    value={voicePreference}
                    onChange={(e) => setVoicePreference(e.target.value as 'auto' | 'egypt' | 'saudi')}
                    className={clsx(
                      'w-full px-3 py-2 rounded-lg border transition-colors',
                      theme === 'dark'
                        ? 'bg-gray-800 border-gray-600 text-white'
                        : 'bg-white border-gray-200 text-gray-900'
                    )}
                  >
                    <option value="auto">{language === 'ar' ? 'تلقائي' : 'Auto'}</option>
                    <option value="egypt">{language === 'ar' ? 'مصري' : 'Egyptian'}</option>
                    <option value="saudi">{language === 'ar' ? 'سعودي' : 'Saudi'}</option>
                  </select>
                </div>
              </div>
              {prefsStatus === 'saving' && (
                <p className="text-xs text-yellow-500 mt-2 flex items-center gap-1">
                  <IconLoader2 size={12} className="animate-spin" /> Saving...
                </p>
              )}
              {prefsStatus === 'saved' && (
                <p className="text-xs text-green-500 mt-2 flex items-center gap-1">
                  <IconCheck size={12} /> Saved
                </p>
              )}
              {prefsError && (
                <p className="text-xs text-red-500 mt-2">{prefsError}</p>
              )}
            </div>
          )}
        </div>

        {/* Transcript Panel */}
        <div className={clsx(
          'rounded-xl border p-4',
          theme === 'dark' ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'
        )}>
          <div className="flex items-center gap-2 mb-4">
            <IconMessage size={18} />
            <h2 className="font-medium">
              {language === 'ar' ? 'المحادثة' : 'Transcript'}
            </h2>
            {isThinking && <IconLoader2 size={14} className="animate-spin text-gray-400" />}
          </div>

          <div className={clsx(
            'h-[400px] lg:h-[500px] overflow-y-auto rounded-lg p-3 space-y-3',
            theme === 'dark' ? 'bg-gray-900/50' : 'bg-gray-50'
          )}>
            {transcript.length === 0 ? (
              <div className="flex items-center justify-center h-full">
                <p className={clsx('text-sm', theme === 'dark' ? 'text-gray-500' : 'text-gray-400')}>
                  {language === 'ar' ? 'ابدأ المكالمة لرؤية المحادثة' : 'Start a call to see transcript'}
                </p>
              </div>
            ) : (
              transcript.map((msg, idx) => (
                <div
                  key={idx}
                  className={clsx(
                    'max-w-[90%] rounded-xl p-3',
                    msg.role === 'user'
                      ? 'bg-blue-600 text-white ml-auto'
                      : msg.role === 'system'
                        ? 'bg-gray-300 dark:bg-gray-700 mx-auto text-center'
                        : 'bg-green-600 text-white'
                  )}
                >
                  <p className="text-sm" dir={language === 'ar' ? 'rtl' : 'ltr'}>{msg.content}</p>
                  <p className="text-xs opacity-70 mt-1">
                    {msg.timestamp.toLocaleTimeString(language === 'ar' ? 'ar-EG' : 'en-US', { 
                      hour: '2-digit', 
                      minute: '2-digit' 
                    })}
                  </p>
                </div>
              ))
            )}
            <div ref={transcriptEndRef} />
          </div>
        </div>
      </div>
      )} {/* end phone call mode */}
    </div>
  );
};

export default VoiceAgentClean;
