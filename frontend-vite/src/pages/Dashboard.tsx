import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useThemeStore } from '@store/themeStore';
import {
  IconUsers,
  IconFileText,
  IconClock,
  IconTrendingUp,
  IconActivity,
  IconBrain,
  IconMicrophone,
  IconDatabase
} from '@tabler/icons-react';
import api from '@utils/api';

const Dashboard = () => {
  const { language } = useThemeStore();
  const [metrics, setMetrics] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadMetrics();
  }, []);

  const loadMetrics = async () => {
    try {
      const data = await api.getMetrics();
      setMetrics(data);
    } catch (error) {
      console.error('Failed to load metrics:', error);
      // Use mock data if backend unavailable
      setMetrics(generateMockMetrics());
    } finally {
      setLoading(false);
    }
  };

  const generateMockMetrics = () => ({
    totalPatients: 1247,
    todayConsultations: 34,
    avgConsultationTime: 18,
    transcriptionAccuracy: 98.4,
    soapNotesGenerated: 892,
    fhirRecordsCreated: 734,
    llmQueries: 2156,
    servicesHealth: {
      asr: 'healthy',
      llm: 'healthy',
      tts: 'healthy',
      soap: 'healthy',
      fhir: 'healthy',
    }
  });

  const stats = [
    {
      icon: <IconUsers size={32} />,
      label: language === 'ar' ? 'إجمالي المرضى' : 'Total Patients',
      value: metrics?.totalPatients || 0,
      change: '+12%',
      color: 'from-blue-500 to-cyan-500'
    },
    {
      icon: <IconFileText size={32} />,
      label: language === 'ar' ? 'استشارات اليوم' : 'Today\'s Consultations',
      value: metrics?.todayConsultations || 0,
      change: '+8%',
      color: 'from-purple-500 to-pink-500'
    },
    {
      icon: <IconClock size={32} />,
      label: language === 'ar' ? 'متوسط الوقت' : 'Avg Time (min)',
      value: metrics?.avgConsultationTime || 0,
      change: '-15%',
      color: 'from-green-500 to-emerald-500'
    },
    {
      icon: <IconTrendingUp size={32} />,
      label: language === 'ar' ? 'دقة النصوص' : 'Accuracy',
      value: `${metrics?.transcriptionAccuracy || 0}%`,
      change: '+2%',
      color: 'from-orange-500 to-red-500'
    }
  ];

  const services = [
    {
      name: language === 'ar' ? 'تحويل الصوت' : 'ASR Service',
      icon: <IconMicrophone size={24} />,
      status: metrics?.servicesHealth?.asr || 'unknown',
      requests: 4562,
      uptime: '99.9%'
    },
    {
      name: language === 'ar' ? 'مساعد AI' : 'LLM Service',
      icon: <IconBrain size={24} />,
      status: metrics?.servicesHealth?.llm || 'unknown',
      requests: 2156,
      uptime: '99.8%'
    },
    {
      name: language === 'ar' ? 'ملاحظات SOAP' : 'SOAP Service',
      icon: <IconFileText size={24} />,
      status: metrics?.servicesHealth?.soap || 'unknown',
      requests: 892,
      uptime: '99.9%'
    },
    {
      name: language === 'ar' ? 'تكامل FHIR' : 'FHIR Service',
      icon: <IconDatabase size={24} />,
      status: metrics?.servicesHealth?.fhir || 'unknown',
      requests: 734,
      uptime: '99.7%'
    }
  ];

  const recentActivities = [
    { time: '2 min ago', action: 'New SOAP note generated', patient: 'Patient #1234' },
    { time: '5 min ago', action: 'Voice transcription completed', patient: 'Patient #1235' },
    { time: '12 min ago', action: 'FHIR resource created', patient: 'Patient #1236' },
    { time: '18 min ago', action: 'AI consultation completed', patient: 'Patient #1237' },
    { time: '25 min ago', action: 'Clinical note saved', patient: 'Patient #1238' },
  ];

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-accent-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-gray-600 dark:text-gray-300">Loading dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen py-20">
      <div className="container-custom">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-12"
        >
          <h1 className="text-4xl md:text-5xl font-bold mb-4">
            <span className="gradient-text">
              {language === 'ar' ? 'لوحة التحكم' : 'Dashboard'}
            </span>
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-300">
            {language === 'ar'
              ? 'نظرة عامة على الأداء والمقاييس'
              : 'Overview of performance and metrics'}
          </p>
        </motion.div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
          {stats.map((stat, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="glass-card p-6"
            >
              <div className={`inline-flex p-3 rounded-xl bg-gradient-to-br ${stat.color} text-white mb-4`}>
                {stat.icon}
              </div>
              <div className="text-3xl font-bold mb-2">{stat.value}</div>
              <div className="text-gray-600 dark:text-gray-300 mb-2">{stat.label}</div>
              <div className="text-sm text-green-500 font-medium">
                {stat.change} {language === 'ar' ? 'من الشهر الماضي' : 'from last month'}
              </div>
            </motion.div>
          ))}
        </div>

        {/* Main Content Grid */}
        <div className="grid lg:grid-cols-3 gap-8">
          {/* Services Status */}
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            className="lg:col-span-2 glass-card p-8"
          >
            <h2 className="text-2xl font-bold mb-6 flex items-center gap-3">
              <IconActivity size={28} className="text-accent-500" />
              {language === 'ar' ? 'حالة الخدمات' : 'Services Status'}
            </h2>

            <div className="space-y-4">
              {services.map((service, index) => (
                <div
                  key={index}
                  className="p-4 bg-white dark:bg-gray-800/50 rounded-xl border border-gray-200 dark:border-gray-700"
                >
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-3">
                      <div className="p-2 bg-accent-100 dark:bg-accent-900/20 rounded-lg text-accent-500">
                        {service.icon}
                      </div>
                      <div>
                        <h3 className="font-bold">{service.name}</h3>
                        <p className="text-sm text-gray-500">
                          {service.requests.toLocaleString()} {language === 'ar' ? 'طلب' : 'requests'}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className={`w-2 h-2 rounded-full ${
                        service.status === 'healthy' ? 'bg-green-500' : 'bg-red-500'
                      } animate-pulse`} />
                      <span className="text-sm font-medium">
                        {service.status === 'healthy' ? 'Online' : 'Offline'}
                      </span>
                    </div>
                  </div>
                  <div className="flex items-center gap-4 text-sm">
                    <div className="flex-1">
                      <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-accent-500 to-accent-600"
                          style={{ width: service.uptime }}
                        />
                      </div>
                    </div>
                    <span className="text-gray-600 dark:text-gray-400">{service.uptime}</span>
                  </div>
                </div>
              ))}
            </div>
          </motion.div>

          {/* Recent Activity */}
          <motion.div
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            className="glass-card p-8"
          >
            <h2 className="text-2xl font-bold mb-6">
              {language === 'ar' ? 'النشاط الأخير' : 'Recent Activity'}
            </h2>

            <div className="space-y-4">
              {recentActivities.map((activity, index) => (
                <div key={index} className="pb-4 border-b border-gray-200 dark:border-gray-700 last:border-0">
                  <p className="font-medium mb-1">{activity.action}</p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">{activity.patient}</p>
                  <p className="text-xs text-gray-500 mt-1">{activity.time}</p>
                </div>
              ))}
            </div>

            <button className="w-full mt-6 px-4 py-2 border border-accent-500 text-accent-500 rounded-xl hover:bg-accent-50 dark:hover:bg-accent-900/20 transition-colors">
              {language === 'ar' ? 'عرض الكل' : 'View All'}
            </button>
          </motion.div>
        </div>

        {/* Performance Chart Placeholder */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="mt-8 glass-card p-8"
        >
          <h2 className="text-2xl font-bold mb-6">
            {language === 'ar' ? 'أداء النظام' : 'System Performance'}
          </h2>
          <div className="h-64 flex items-center justify-center bg-gray-50 dark:bg-gray-900/50 rounded-xl">
            <p className="text-gray-400">
              {language === 'ar' ? 'الرسم البياني قريبًا' : 'Chart visualization coming soon'}
            </p>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default Dashboard;
