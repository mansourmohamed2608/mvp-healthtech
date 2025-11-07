import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useThemeStore } from '@store/themeStore';
import {
  IconPhone,
  IconPhoneOff,
  IconMicrophone,
  IconMicrophoneOff,
  IconPlayerPlay,
  IconPlayerStop,
  IconCheck,
  IconAlertCircle
} from '@tabler/icons-react';
import { Device, Call } from '@twilio/voice-sdk';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

const VoiceAgent = () => {
  const { language } = useThemeStore();
  const [device, setDevice] = useState<Device | null>(null);
  const [call, setCall] = useState<Call | null>(null);
  const [callStatus, setCallStatus] = useState<'idle' | 'connecting' | 'connected' | 'disconnected'>('idle');
  const [transcript, setTranscript] = useState<Message[]>([]);
  const [error, setError] = useState<string>('');
  const [isMuted, setIsMuted] = useState(false);
  const [deviceReady, setDeviceReady] = useState(false);
  const transcriptEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll transcript to bottom
  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [transcript]);

  // Initialize Twilio Device on mount
  useEffect(() => {
    async function initTwilioDevice() {
      try {
        // Try to get token from gateway (POST /twilio/token)
        const response = await fetch('http://localhost:3001/twilio/token', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
        });
        if (!response.ok) {
          throw new Error('Failed to fetch Twilio token from gateway');
        }
        const { token } = await response.json();

        const twilioDevice = new Device(token, {
          codecPreferences: [Call.Codec.Opus, Call.Codec.PCMU],
          enableRingingState: true,
          edge: 'ashburn',
        });

        twilioDevice.on('registered', () => {
          console.log('✅ Twilio Device registered and ready');
          setDeviceReady(true);
          setError('');
        });

        twilioDevice.on('error', (error: any) => {
          console.error('❌ Twilio Device error:', error);
          setError(`Device error: ${error.message}`);
          setDeviceReady(false);
        });

        twilioDevice.on('unregistered', () => {
          console.log('Device unregistered');
          setDeviceReady(false);
        });

        await twilioDevice.register();
        setDevice(twilioDevice);
      } catch (err: any) {
        console.error('Failed to initialize Twilio Device:', err);
        setError(`Initialization failed: ${err.message}. Make sure gateway is running on localhost:3001`);
        setDeviceReady(false);
      }
    }

    initTwilioDevice();

    // Cleanup on unmount
    return () => {
      if (device) {
        device.destroy();
      }
    };
  }, []);

  const startCall = async () => {
    if (!device) {
      setError('Device not initialized');
      return;
    }

    try {
      setCallStatus('connecting');
      setError('');
      setTranscript([]);

      const params = {
        To: '+1234567890', // Placeholder
      };

      const outgoingCall = await device.connect({ params });
      setCall(outgoingCall);

      outgoingCall.on('accept', () => {
        console.log('✅ Call connected');
        setCallStatus('connected');
        setTranscript([
          {
            role: 'assistant',
            content: language === 'ar' 
              ? 'مرحبا بك في النظام الطبي الذكي. كيف يمكنني مساعدتك؟'
              : 'Welcome to the smart medical assistant. How can I help you?',
            timestamp: new Date(),
          },
        ]);
      });

      outgoingCall.on('disconnect', () => {
        console.log('Call disconnected');
        setCallStatus('disconnected');
        setCall(null);
        setTimeout(() => setCallStatus('idle'), 2000);
      });

      outgoingCall.on('cancel', () => {
        console.log('Call cancelled');
        setCallStatus('idle');
        setCall(null);
      });

      outgoingCall.on('error', (error: any) => {
        console.error('Call error:', error);
        setError(`Call error: ${error.message}`);
        setCallStatus('idle');
        setCall(null);
      });
    } catch (err: any) {
      console.error('Failed to start call:', err);
      setError(`Failed to connect: ${err.message}`);
      setCallStatus('idle');
    }
  };

  const endCall = () => {
    if (call) {
      call.disconnect();
      setCall(null);
      setCallStatus('idle');
    }
  };

  const toggleMute = () => {
    if (call) {
      call.mute(!isMuted);
      setIsMuted(!isMuted);
    }
  };

  const getStatusColor = () => {
    switch (callStatus) {
      case 'connected': return 'from-green-500 to-emerald-500';
      case 'connecting': return 'from-yellow-500 to-orange-500';
      case 'disconnected': return 'from-gray-500 to-slate-500';
      default: return 'from-blue-500 to-indigo-500';
    }
  };

  const getStatusText = () => {
    if (language === 'ar') {
      switch (callStatus) {
        case 'idle': return 'جاهز للاتصال';
        case 'connecting': return 'جاري الاتصال...';
        case 'connected': return 'متصل';
        case 'disconnected': return 'تم إنهاء المكالمة';
      }
    } else {
      switch (callStatus) {
        case 'idle': return 'Ready to Call';
        case 'connecting': return 'Connecting...';
        case 'connected': return 'Connected';
        case 'disconnected': return 'Call Ended';
      }
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 dark:from-slate-900 dark:via-slate-800 dark:to-indigo-950 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-6xl mx-auto space-y-8">
        
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center"
        >
          <h1 className="text-5xl font-bold bg-gradient-to-r from-accent via-blue-600 to-indigo-600 bg-clip-text text-transparent mb-4">
            {language === 'ar' ? 'مساعد طبي صوتي ذكي' : 'AI Voice Medical Assistant'}
          </h1>
          <p className="text-xl text-slate-600 dark:text-slate-400">
            {language === 'ar' 
              ? 'تحدث مع المساعد الطبي الذكي عبر الاتصال الصوتي المباشر'
              : 'Talk to the AI medical assistant via real-time voice call'
            }
          </p>
        </motion.div>

        {/* Status Card */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="glass-card p-8 rounded-3xl"
        >
          <div className="flex items-center justify-between mb-8">
            <div className="flex items-center gap-4">
              <motion.div
                animate={{
                  scale: callStatus === 'connected' || callStatus === 'connecting' ? [1, 1.2, 1] : 1,
                }}
                transition={{ repeat: callStatus === 'connected' || callStatus === 'connecting' ? Infinity : 0, duration: 1.5 }}
                className={`w-6 h-6 rounded-full bg-gradient-to-r ${getStatusColor()}`}
              />
              <span className="text-2xl font-semibold text-slate-800 dark:text-white">
                {getStatusText()}
              </span>
            </div>
            
            {deviceReady && (
              <div className="flex items-center gap-2 text-green-600 dark:text-green-400">
                <IconCheck size={20} />
                <span className="text-sm font-medium">
                  {language === 'ar' ? 'الجهاز جاهز' : 'Device Ready'}
                </span>
              </div>
            )}
          </div>

          {/* Error Display */}
          <AnimatePresence>
            {error && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="mb-6 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl flex items-start gap-3"
              >
                <IconAlertCircle size={24} className="text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" />
                <p className="text-red-700 dark:text-red-300 text-sm leading-relaxed">{error}</p>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Control Buttons */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {callStatus === 'idle' && (
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={startCall}
                disabled={!deviceReady}
                className="col-span-full flex items-center justify-center gap-3 bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700 disabled:from-slate-300 disabled:to-slate-400 disabled:cursor-not-allowed text-white font-bold py-5 px-8 rounded-2xl transition-all duration-200 text-lg shadow-lg shadow-green-500/30"
              >
                <IconPhone size={24} />
                {language === 'ar' ? 'ابدأ المحادثة' : 'Start Call'}
              </motion.button>
            )}

            {(callStatus === 'connected' || callStatus === 'connecting') && (
              <>
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={toggleMute}
                  className={`flex items-center justify-center gap-3 ${
                    isMuted 
                      ? 'bg-gradient-to-r from-yellow-500 to-orange-500 hover:from-yellow-600 hover:to-orange-600' 
                      : 'bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700'
                  } text-white font-bold py-5 px-8 rounded-2xl transition-all duration-200 text-lg shadow-lg`}
                >
                  {isMuted ? <IconMicrophoneOff size={24} /> : <IconMicrophone size={24} />}
                  {language === 'ar' ? (isMuted ? 'إلغاء الكتم' : 'كتم الصوت') : (isMuted ? 'Unmute' : 'Mute')}
                </motion.button>
                
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={endCall}
                  className="md:col-span-2 flex items-center justify-center gap-3 bg-gradient-to-r from-red-500 to-rose-600 hover:from-red-600 hover:to-rose-700 text-white font-bold py-5 px-8 rounded-2xl transition-all duration-200 text-lg shadow-lg shadow-red-500/30"
                >
                  <IconPhoneOff size={24} />
                  {language === 'ar' ? 'إنهاء المكالمة' : 'End Call'}
                </motion.button>
              </>
            )}
          </div>
        </motion.div>

        {/* Transcript Card */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="glass-card p-8 rounded-3xl"
        >
          <h2 className="text-3xl font-bold text-slate-800 dark:text-white mb-6 flex items-center gap-3">
            <IconPlayerPlay size={32} className="text-accent" />
            {language === 'ar' ? 'نص المحادثة' : 'Live Transcript'}
          </h2>
          
          <div className="h-[500px] overflow-y-auto bg-slate-100 dark:bg-slate-800/50 rounded-2xl p-6 space-y-4 custom-scrollbar">
            {transcript.length === 0 ? (
              <div className="flex items-center justify-center h-full">
                <div className="text-center">
                  <IconPlayerStop size={64} className="mx-auto text-slate-400 dark:text-slate-600 mb-4" />
                  <p className="text-slate-500 dark:text-slate-400 text-lg">
                    {language === 'ar' 
                      ? 'ابدأ المحادثة لرؤية النص المباشر هنا'
                      : 'Start the call to see the live transcript here'
                    }
                  </p>
                </div>
              </div>
            ) : (
              <AnimatePresence>
                {transcript.map((msg, idx) => (
                  <motion.div
                    key={idx}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    className={`flex ${msg.role === 'user' ? 'justify-start' : 'justify-end'}`}
                  >
                    <div
                      className={`max-w-[80%] rounded-2xl p-5 shadow-lg ${
                        msg.role === 'user'
                          ? 'bg-gradient-to-br from-blue-500 to-indigo-600 text-white'
                          : 'bg-gradient-to-br from-green-500 to-emerald-600 text-white'
                      }`}
                    >
                      <p className="text-lg leading-relaxed mb-2" dir={language === 'ar' ? 'rtl' : 'ltr'}>
                        {msg.content}
                      </p>
                      <p className="text-xs opacity-75">
                        {msg.timestamp.toLocaleTimeString(language === 'ar' ? 'ar-EG' : 'en-US')}
                      </p>
                    </div>
                  </motion.div>
                ))}
              </AnimatePresence>
            )}
            <div ref={transcriptEndRef} />
          </div>
        </motion.div>

        {/* Instructions */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 }}
          className="glass-card p-6 rounded-2xl border-l-4 border-accent"
        >
          <h3 className="text-lg font-semibold text-slate-800 dark:text-white mb-3">
            {language === 'ar' ? 'كيفية الاستخدام:' : 'How to Use:'}
          </h3>
          <ul className="list-disc list-inside text-slate-600 dark:text-slate-300 space-y-2" dir={language === 'ar' ? 'rtl' : 'ltr'}>
            {language === 'ar' ? (
              <>
                <li>تأكد من تشغيل Gateway على المنفذ 3001</li>
                <li>انقر على "ابدأ المحادثة" للاتصال بالمساعد الطبي</li>
                <li>تحدث بوضوح باللغة العربية أو الإنجليزية</li>
                <li>سيظهر نص المحادثة تلقائياً في الوقت الفعلي</li>
                <li>يمكنك كتم الصوت أو إنهاء المكالمة في أي وقت</li>
              </>
            ) : (
              <>
                <li>Make sure Gateway is running on port 3001</li>
                <li>Click "Start Call" to connect to the medical assistant</li>
                <li>Speak clearly in Arabic or English</li>
                <li>The transcript will appear automatically in real-time</li>
                <li>You can mute or end the call anytime</li>
              </>
            )}
          </ul>
        </motion.div>
      </div>
    </div>
  );
};

export default VoiceAgent;
