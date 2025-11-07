import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { useThemeStore } from '@store/themeStore';
import {
  IconMicrophone,
  IconBrain,
  IconFileText,
  IconDatabase,
  IconStethoscope,
  IconChartBar,
  IconArrowRight,
  IconSparkles,
  IconClock,
  IconShieldCheck
} from '@tabler/icons-react';

const Features = () => {
  const { language } = useThemeStore();
  const isRTL = language === 'ar';

  const features = [
    {
      icon: <IconMicrophone size={40} />,
      title: language === 'ar' ? 'تحويل الصوت إلى نص' : 'Voice Transcription',
      description: language === 'ar'
        ? 'تحويل المحادثات الطبية إلى نص دقيق باستخدام الذكاء الاصطناعي المتقدم ودعم اللهجات العربية'
        : 'Convert medical conversations to accurate text using advanced AI with Arabic dialect support',
      link: '/features/voice-transcription',
      gradient: 'from-blue-500 to-cyan-500',
      stats: [
        { label: language === 'ar' ? 'دقة' : 'Accuracy', value: '98%' },
        { label: language === 'ar' ? 'لهجات' : 'Dialects', value: '5+' },
      ]
    },
    {
      icon: <IconFileText size={40} />,
      title: language === 'ar' ? 'ملاحظات SOAP' : 'SOAP Notes',
      description: language === 'ar'
        ? 'توليد ملاحظات SOAP منظمة تلقائيًا من نصوص الاستشارات الطبية'
        : 'Auto-generate structured SOAP notes from medical consultation transcripts',
      link: '/features/soap-generation',
      gradient: 'from-purple-500 to-pink-500',
      stats: [
        { label: language === 'ar' ? 'وقت' : 'Time Saved', value: '75%' },
        { label: language === 'ar' ? 'ملاحظات' : 'Notes/Day', value: '50+' },
      ]
    },
    {
      icon: <IconDatabase size={40} />,
      title: language === 'ar' ? 'تكامل FHIR' : 'FHIR Integration',
      description: language === 'ar'
        ? 'تكامل سلس مع أنظمة السجلات الصحية الإلكترونية باستخدام معيار FHIR'
        : 'Seamless integration with EHR systems using FHIR standard',
      link: '/features/fhir-integration',
      gradient: 'from-green-500 to-emerald-500',
      stats: [
        { label: language === 'ar' ? 'متوافق' : 'Compatible', value: '100%' },
        { label: language === 'ar' ? 'موارد' : 'Resources', value: '20+' },
      ]
    },
    {
      icon: <IconStethoscope size={40} />,
      title: language === 'ar' ? 'ملاحظات سريرية' : 'Clinical Notes',
      description: language === 'ar'
        ? 'إدارة وتنظيم الملاحظات السريرية مع محرك بحث متقدم'
        : 'Manage and organize clinical notes with advanced search',
      link: '/features/clinical-notes',
      gradient: 'from-orange-500 to-red-500',
      stats: [
        { label: language === 'ar' ? 'بحث' : 'Search', value: '<1s' },
        { label: language === 'ar' ? 'تصدير' : 'Export', value: 'PDF/JSON' },
      ]
    },
    {
      icon: <IconBrain size={40} />,
      title: language === 'ar' ? 'مساعد ذكي' : 'AI Assistant',
      description: language === 'ar'
        ? 'مساعد طبي ذكي يفهم السياق ويقدم اقتراحات مدعومة بالأدلة'
        : 'Context-aware medical AI assistant with evidence-based suggestions',
      link: '/dashboard',
      gradient: 'from-indigo-500 to-purple-500',
      stats: [
        { label: language === 'ar' ? 'استجابة' : 'Response', value: '<2s' },
        { label: language === 'ar' ? 'لغات' : 'Languages', value: '3+' },
      ]
    },
    {
      icon: <IconChartBar size={40} />,
      title: language === 'ar' ? 'التحليلات' : 'Analytics',
      description: language === 'ar'
        ? 'رؤى شاملة عن أداء العيادة ومقاييس المرضى'
        : 'Comprehensive insights into clinic performance and patient metrics',
      link: '/dashboard',
      gradient: 'from-yellow-500 to-orange-500',
      stats: [
        { label: language === 'ar' ? 'لوحات' : 'Dashboards', value: '5+' },
        { label: language === 'ar' ? 'تقارير' : 'Reports', value: 'Real-time' },
      ]
    },
  ];

  const benefits = [
    {
      icon: <IconClock size={32} />,
      title: language === 'ar' ? 'توفير الوقت' : 'Save Time',
      description: language === 'ar'
        ? 'تقليل وقت التوثيق بنسبة 75٪ مع الأتمتة الذكية'
        : 'Reduce documentation time by 75% with intelligent automation',
    },
    {
      icon: <IconShieldCheck size={32} />,
      title: language === 'ar' ? 'آمن ومتوافق' : 'Secure & Compliant',
      description: language === 'ar'
        ? 'متوافق مع HIPAA مع تشفير من الدرجة العسكرية'
        : 'HIPAA compliant with military-grade encryption',
    },
    {
      icon: <IconSparkles size={32} />,
      title: language === 'ar' ? 'دقة عالية' : 'High Accuracy',
      description: language === 'ar'
        ? 'دقة 98٪+ في النصوص الطبية مع التعلم المستمر'
        : '98%+ accuracy in medical transcriptions with continuous learning',
    },
  ];

  return (
    <div className="min-h-screen py-20">
      {/* Hero Section */}
      <section className="container-custom py-16">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-center max-w-4xl mx-auto"
        >
          <h1 className="text-5xl md:text-7xl font-bold mb-6 kinetic-text">
            <span className="gradient-text">
              {language === 'ar' ? 'ميزات قوية' : 'Powerful Features'}
            </span>
          </h1>
          <p className="text-xl md:text-2xl text-gray-600 dark:text-gray-300 mb-8">
            {language === 'ar'
              ? 'كل ما تحتاجه لتحويل ممارستك الطبية إلى العصر الرقمي'
              : 'Everything you need to transform your medical practice into the digital age'}
          </p>
        </motion.div>
      </section>

      {/* Features Grid */}
      <section className="container-custom py-16">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
            >
              <Link to={feature.link} className="block h-full">
                <div className="glass-card h-full p-8 group hover:scale-[1.02] transition-transform duration-300">
                  <div className={`inline-flex p-4 rounded-2xl bg-gradient-to-br ${feature.gradient} text-white mb-6`}>
                    {feature.icon}
                  </div>

                  <h3 className="text-2xl font-bold mb-4">
                    {feature.title}
                  </h3>

                  <p className="text-gray-600 dark:text-gray-300 mb-6">
                    {feature.description}
                  </p>

                  <div className="grid grid-cols-2 gap-4 mb-6">
                    {feature.stats.map((stat, i) => (
                      <div key={i}>
                        <div className="text-3xl font-bold gradient-text">{stat.value}</div>
                        <div className="text-sm text-gray-500">{stat.label}</div>
                      </div>
                    ))}
                  </div>

                  <div className="flex items-center gap-2 text-accent-500 font-semibold group-hover:gap-4 transition-all">
                    {language === 'ar' ? 'اعرف المزيد' : 'Learn More'}
                    <IconArrowRight size={20} className={isRTL ? 'rotate-180' : ''} />
                  </div>
                </div>
              </Link>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Benefits Section */}
      <section className="container-custom py-16">
        <motion.h2
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          className="text-4xl md:text-5xl font-bold text-center mb-16"
        >
          {language === 'ar' ? 'لماذا نحن؟' : 'Why Choose Us?'}
        </motion.h2>

        <div className="grid md:grid-cols-3 gap-8">
          {benefits.map((benefit, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
              className="glass-card p-8 text-center"
            >
              <div className="inline-flex p-4 rounded-full bg-accent-100 dark:bg-accent-900/20 text-accent-500 mb-6">
                {benefit.icon}
              </div>
              <h3 className="text-2xl font-bold mb-4">{benefit.title}</h3>
              <p className="text-gray-600 dark:text-gray-300">{benefit.description}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* CTA Section */}
      <section className="container-custom py-16">
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
          className="glass-card p-12 text-center"
        >
          <h2 className="text-4xl font-bold mb-6">
            {language === 'ar' ? 'جاهز للبدء؟' : 'Ready to Get Started?'}
          </h2>
          <p className="text-xl text-gray-600 dark:text-gray-300 mb-8">
            {language === 'ar'
              ? 'جرب النظام الآن واكتشف كيف يمكننا تحسين ممارستك الطبية'
              : 'Try our platform today and see how we can improve your medical practice'}
          </p>
          <Link to="/demo">
            <button className="magnetic-btn">
              {language === 'ar' ? 'احجز عرضًا توضيحيًا' : 'Book a Demo'}
            </button>
          </Link>
        </motion.div>
      </section>
    </div>
  );
};

export default Features;
