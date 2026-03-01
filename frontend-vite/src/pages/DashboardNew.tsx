import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useThemeStore } from '@store/themeStore';
import { useAuthStore } from '@store/authStore';
import {
  IconPhone,
  IconFileText,
  IconClock,
  IconCheck,
  IconAlertCircle,
  IconArrowRight,
  IconRefresh,
  IconActivity,
  IconUsers,
  IconCalendar
} from '@tabler/icons-react';
import clsx from 'clsx';

interface Session {
  id: string;
  patientName?: string;
  startTime: Date;
  status: 'active' | 'completed' | 'pending';
  duration?: number;
}

interface DashboardStats {
  activeSessions: number;
  completedToday: number;
  pendingNotes: number;
  totalPatients: number;
}

const Dashboard = () => {
  const { theme, language } = useThemeStore();
  const { userId } = useAuthStore();
  const [stats] = useState<DashboardStats>({
    activeSessions: 0,
    completedToday: 0,
    pendingNotes: 0,
    totalPatients: 0
  });
  const [recentSessions] = useState<Session[]>([]);
  const [servicesStatus, setServicesStatus] = useState<Record<string, boolean>>({});

  useEffect(() => {
    checkServices();
  }, []);

  const checkServices = async () => {
    const services = {
      'Auth Service': '/api/auth/health',
      'SOAP Service': '/api/soap/health',
      'TTS Service': '/api/tts/health',
      'LLM Service': '/api/llm/health'
    };

    const results: Record<string, boolean> = {};
    for (const [name, path] of Object.entries(services)) {
      try {
        const response = await fetch(path);
        results[name] = response.ok;
      } catch {
        results[name] = false;
      }
    }
    setServicesStatus(results);
  };

  const StatCard = ({ 
    title, 
    value, 
    icon: Icon, 
    color 
  }: { 
    title: string; 
    value: number | string; 
    icon: any; 
    color: string;
  }) => (
    <div className={clsx(
      'rounded-xl p-5 border',
      theme === 'dark' 
        ? 'bg-gray-800 border-gray-700' 
        : 'bg-white border-gray-200'
    )}>
      <div className="flex items-start justify-between">
        <div>
          <p className={clsx(
            'text-sm font-medium',
            theme === 'dark' ? 'text-gray-400' : 'text-gray-500'
          )}>
            {title}
          </p>
          <p className="text-3xl font-bold mt-1">{value}</p>
        </div>
        <div className={clsx(
          'w-12 h-12 rounded-lg flex items-center justify-center',
          color
        )}>
          <Icon size={24} className="text-white" />
        </div>
      </div>
    </div>
  );

  const ServiceStatus = ({ name, healthy }: { name: string; healthy: boolean }) => (
    <div className={clsx(
      'flex items-center justify-between py-2 px-3 rounded-lg',
      theme === 'dark' ? 'bg-gray-700/50' : 'bg-gray-50'
    )}>
      <span className="text-sm">{name}</span>
      <span className={clsx(
        'flex items-center gap-1 text-xs font-medium',
        healthy ? 'text-green-500' : 'text-red-500'
      )}>
        {healthy ? <IconCheck size={14} /> : <IconAlertCircle size={14} />}
        {healthy ? 'Healthy' : 'Down'}
      </span>
    </div>
  );

  return (
    <div className="max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl lg:text-3xl font-bold">
          {language === 'ar' ? 'لوحة التحكم' : 'Dashboard'}
        </h1>
        <p className={clsx(
          'mt-1',
          theme === 'dark' ? 'text-gray-400' : 'text-gray-500'
        )}>
          {language === 'ar' 
            ? `مرحباً، ${userId || 'مستخدم'}` 
            : `Welcome back, ${userId || 'User'}`}
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <StatCard
          title={language === 'ar' ? 'جلسات نشطة' : 'Active Sessions'}
          value={stats.activeSessions}
          icon={IconActivity}
          color="bg-blue-600"
        />
        <StatCard
          title={language === 'ar' ? 'مكتمل اليوم' : 'Completed Today'}
          value={stats.completedToday}
          icon={IconCheck}
          color="bg-green-600"
        />
        <StatCard
          title={language === 'ar' ? 'ملاحظات معلقة' : 'Pending Notes'}
          value={stats.pendingNotes}
          icon={IconFileText}
          color="bg-yellow-600"
        />
        <StatCard
          title={language === 'ar' ? 'إجمالي المرضى' : 'Total Patients'}
          value={stats.totalPatients}
          icon={IconUsers}
          color="bg-purple-600"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Quick Actions */}
        <div className={clsx(
          'lg:col-span-2 rounded-xl border p-6',
          theme === 'dark' 
            ? 'bg-gray-800 border-gray-700' 
            : 'bg-white border-gray-200'
        )}>
          <h2 className="text-lg font-semibold mb-4">
            {language === 'ar' ? 'إجراءات سريعة' : 'Quick Actions'}
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <Link
              to="/voice-agent"
              className={clsx(
                'flex items-center gap-4 p-4 rounded-xl border transition-all',
                theme === 'dark'
                  ? 'border-gray-700 hover:border-blue-500 hover:bg-blue-500/10'
                  : 'border-gray-200 hover:border-blue-500 hover:bg-blue-50'
              )}
            >
              <div className="w-12 h-12 rounded-xl bg-blue-600 flex items-center justify-center flex-shrink-0">
                <IconPhone size={24} className="text-white" />
              </div>
              <div className="flex-1 min-w-0">
                <h3 className="font-medium">
                  {language === 'ar' ? 'بدء جلسة صوتية' : 'Start Voice Session'}
                </h3>
                <p className={clsx(
                  'text-sm truncate',
                  theme === 'dark' ? 'text-gray-400' : 'text-gray-500'
                )}>
                  {language === 'ar' 
                    ? 'محادثة صوتية مع المريض' 
                    : 'Begin a voice consultation'}
                </p>
              </div>
              <IconArrowRight size={20} className={theme === 'dark' ? 'text-gray-500' : 'text-gray-400'} />
            </Link>

            <Link
              to="/clinical-notes"
              className={clsx(
                'flex items-center gap-4 p-4 rounded-xl border transition-all',
                theme === 'dark'
                  ? 'border-gray-700 hover:border-green-500 hover:bg-green-500/10'
                  : 'border-gray-200 hover:border-green-500 hover:bg-green-50'
              )}
            >
              <div className="w-12 h-12 rounded-xl bg-green-600 flex items-center justify-center flex-shrink-0">
                <IconFileText size={24} className="text-white" />
              </div>
              <div className="flex-1 min-w-0">
                <h3 className="font-medium">
                  {language === 'ar' ? 'تسجيل ملاحظات' : 'Record Clinical Notes'}
                </h3>
                <p className={clsx(
                  'text-sm truncate',
                  theme === 'dark' ? 'text-gray-400' : 'text-gray-500'
                )}>
                  {language === 'ar' 
                    ? 'تحويل الصوت إلى SOAP' 
                    : 'Convert audio to SOAP notes'}
                </p>
              </div>
              <IconArrowRight size={20} className={theme === 'dark' ? 'text-gray-500' : 'text-gray-400'} />
            </Link>
          </div>

          {/* Recent Sessions */}
          <h3 className="text-lg font-semibold mt-8 mb-4">
            {language === 'ar' ? 'الجلسات الأخيرة' : 'Recent Sessions'}
          </h3>
          {recentSessions.length === 0 ? (
            <div className={clsx(
              'text-center py-12 rounded-xl border-2 border-dashed',
              theme === 'dark' ? 'border-gray-700' : 'border-gray-200'
            )}>
              <IconClock size={48} className={clsx(
                'mx-auto mb-3',
                theme === 'dark' ? 'text-gray-600' : 'text-gray-300'
              )} />
              <p className={theme === 'dark' ? 'text-gray-400' : 'text-gray-500'}>
                {language === 'ar' 
                  ? 'لا توجد جلسات حديثة' 
                  : 'No recent sessions'}
              </p>
              <Link
                to="/voice-agent"
                className="inline-flex items-center gap-2 mt-4 text-blue-500 hover:text-blue-600"
              >
                {language === 'ar' ? 'بدء جلسة جديدة' : 'Start a new session'}
                <IconArrowRight size={16} />
              </Link>
            </div>
          ) : (
            <div className="space-y-2">
              {recentSessions.map((session) => (
                <div
                  key={session.id}
                  className={clsx(
                    'flex items-center justify-between p-3 rounded-lg',
                    theme === 'dark' ? 'bg-gray-700/50' : 'bg-gray-50'
                  )}
                >
                  <div className="flex items-center gap-3">
                    <div className={clsx(
                      'w-2 h-2 rounded-full',
                      session.status === 'active' ? 'bg-green-500' :
                      session.status === 'completed' ? 'bg-blue-500' : 'bg-yellow-500'
                    )} />
                    <div>
                      <p className="font-medium">{session.patientName || 'Patient'}</p>
                      <p className={clsx(
                        'text-xs',
                        theme === 'dark' ? 'text-gray-400' : 'text-gray-500'
                      )}>
                        {session.startTime.toLocaleTimeString()}
                      </p>
                    </div>
                  </div>
                  <span className={clsx(
                    'text-xs font-medium px-2 py-1 rounded-full',
                    session.status === 'active' 
                      ? 'bg-green-500/20 text-green-500'
                      : session.status === 'completed'
                        ? 'bg-blue-500/20 text-blue-500'
                        : 'bg-yellow-500/20 text-yellow-500'
                  )}>
                    {session.status}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Service Status */}
        <div className={clsx(
          'rounded-xl border p-6',
          theme === 'dark' 
            ? 'bg-gray-800 border-gray-700' 
            : 'bg-white border-gray-200'
        )}>
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold">
              {language === 'ar' ? 'حالة الخدمات' : 'Service Status'}
            </h2>
            <button
              onClick={checkServices}
              className={clsx(
                'p-2 rounded-lg transition-colors',
                theme === 'dark' ? 'hover:bg-gray-700' : 'hover:bg-gray-100'
              )}
              title="Refresh"
            >
              <IconRefresh size={18} />
            </button>
          </div>
          <div className="space-y-2">
            {Object.entries(servicesStatus).map(([name, healthy]) => (
              <ServiceStatus key={name} name={name} healthy={healthy} />
            ))}
            {Object.keys(servicesStatus).length === 0 && (
              <p className={clsx(
                'text-sm text-center py-4',
                theme === 'dark' ? 'text-gray-400' : 'text-gray-500'
              )}>
                {language === 'ar' ? 'جاري التحقق...' : 'Checking services...'}
              </p>
            )}
          </div>

          {/* Today's Date */}
          <div className={clsx(
            'mt-6 p-4 rounded-xl',
            theme === 'dark' ? 'bg-gray-700/50' : 'bg-gray-50'
          )}>
            <div className="flex items-center gap-3">
              <IconCalendar size={20} className="text-blue-500" />
              <div>
                <p className={clsx(
                  'text-xs',
                  theme === 'dark' ? 'text-gray-400' : 'text-gray-500'
                )}>
                  {language === 'ar' ? 'اليوم' : 'Today'}
                </p>
                <p className="font-medium">
                  {new Date().toLocaleDateString(language === 'ar' ? 'ar-EG' : 'en-US', {
                    weekday: 'long',
                    year: 'numeric',
                    month: 'long',
                    day: 'numeric'
                  })}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
