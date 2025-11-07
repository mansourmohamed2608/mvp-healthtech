import { useState } from 'react';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { useThemeStore } from '@store/themeStore';
import { IconCheck, IconX, IconSparkles } from '@tabler/icons-react';

const Pricing = () => {
  const { language } = useThemeStore();
  const [billingPeriod, setBillingPeriod] = useState<'monthly' | 'annual'>('monthly');

  const plans = [
    {
      name: language === 'ar' ? 'المبتدئ' : 'Starter',
      description: language === 'ar' ? 'للأطباء الأفراد' : 'For individual practitioners',
      monthlyPrice: 99,
      annualPrice: 990,
      features: [
        { text: language === 'ar' ? '100 استشارة شهريًا' : '100 consultations/month', included: true },
        { text: language === 'ar' ? 'تحويل الصوت إلى نص' : 'Voice transcription', included: true },
        { text: language === 'ar' ? 'ملاحظات SOAP' : 'SOAP notes generation', included: true },
        { text: language === 'ar' ? 'تكامل FHIR الأساسي' : 'Basic FHIR integration', included: true },
        { text: language === 'ar' ? 'مساعد AI' : 'AI assistant', included: false },
        { text: language === 'ar' ? 'لوحة التحكم المتقدمة' : 'Advanced analytics', included: false },
        { text: language === 'ar' ? 'دعم أولوية' : 'Priority support', included: false },
      ],
      popular: false,
      gradient: 'from-blue-500 to-cyan-500',
    },
    {
      name: language === 'ar' ? 'المحترف' : 'Professional',
      description: language === 'ar' ? 'للعيادات الصغيرة' : 'For small clinics',
      monthlyPrice: 299,
      annualPrice: 2990,
      features: [
        { text: language === 'ar' ? '500 استشارة شهريًا' : '500 consultations/month', included: true },
        { text: language === 'ar' ? 'تحويل الصوت إلى نص' : 'Voice transcription', included: true },
        { text: language === 'ar' ? 'ملاحظات SOAP' : 'SOAP notes generation', included: true },
        { text: language === 'ar' ? 'تكامل FHIR كامل' : 'Full FHIR integration', included: true },
        { text: language === 'ar' ? 'مساعد AI' : 'AI assistant', included: true },
        { text: language === 'ar' ? 'لوحة التحكم المتقدمة' : 'Advanced analytics', included: true },
        { text: language === 'ar' ? 'دعم أولوية' : 'Priority support', included: false },
      ],
      popular: true,
      gradient: 'from-purple-500 to-pink-500',
    },
    {
      name: language === 'ar' ? 'المؤسسات' : 'Enterprise',
      description: language === 'ar' ? 'للمستشفيات والمراكز الكبيرة' : 'For hospitals & large practices',
      monthlyPrice: 999,
      annualPrice: 9990,
      features: [
        { text: language === 'ar' ? 'استشارات غير محدودة' : 'Unlimited consultations', included: true },
        { text: language === 'ar' ? 'تحويل الصوت إلى نص' : 'Voice transcription', included: true },
        { text: language === 'ar' ? 'ملاحظات SOAP' : 'SOAP notes generation', included: true },
        { text: language === 'ar' ? 'تكامل FHIR مخصص' : 'Custom FHIR integration', included: true },
        { text: language === 'ar' ? 'مساعد AI متقدم' : 'Advanced AI assistant', included: true },
        { text: language === 'ar' ? 'تحليلات شاملة' : 'Comprehensive analytics', included: true },
        { text: language === 'ar' ? 'دعم 24/7 مخصص' : 'Dedicated 24/7 support', included: true },
      ],
      popular: false,
      gradient: 'from-orange-500 to-red-500',
    },
  ];

  const faqs = [
    {
      question: language === 'ar' ? 'هل يمكنني تغيير خطتي لاحقًا؟' : 'Can I change my plan later?',
      answer: language === 'ar'
        ? 'نعم، يمكنك الترقية أو التخفيض في أي وقت. سيتم تعديل الفاتورة بشكل تناسبي.'
        : 'Yes, you can upgrade or downgrade at any time. Billing will be prorated accordingly.',
    },
    {
      question: language === 'ar' ? 'هل البيانات آمنة؟' : 'Is my data secure?',
      answer: language === 'ar'
        ? 'نحن متوافقون مع HIPAA ونستخدم تشفير من الدرجة العسكرية لحماية جميع البيانات الطبية.'
        : 'We are HIPAA compliant and use military-grade encryption to protect all medical data.',
    },
    {
      question: language === 'ar' ? 'هل هناك عقد طويل الأجل؟' : 'Is there a long-term contract?',
      answer: language === 'ar'
        ? 'لا، جميع الخطط شهرية أو سنوية بدون التزام طويل الأجل. يمكنك الإلغاء في أي وقت.'
        : 'No, all plans are month-to-month or annual with no long-term commitment. Cancel anytime.',
    },
    {
      question: language === 'ar' ? 'هل تقدمون تجربة مجانية؟' : 'Do you offer a free trial?',
      answer: language === 'ar'
        ? 'نعم، نقدم تجربة مجانية لمدة 14 يومًا لجميع الخطط دون الحاجة إلى بطاقة ائتمان.'
        : 'Yes, we offer a 14-day free trial for all plans with no credit card required.',
    },
  ];

  return (
    <div className="min-h-screen py-20">
      {/* Hero Section */}
      <section className="container-custom py-16">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center max-w-4xl mx-auto"
        >
          <h1 className="text-5xl md:text-7xl font-bold mb-6 kinetic-text">
            <span className="gradient-text">
              {language === 'ar' ? 'الأسعار' : 'Pricing Plans'}
            </span>
          </h1>
          <p className="text-xl md:text-2xl text-gray-600 dark:text-gray-300 mb-8">
            {language === 'ar'
              ? 'اختر الخطة المناسبة لممارستك الطبية'
              : 'Choose the perfect plan for your practice'}
          </p>

          {/* Billing Toggle */}
          <div className="inline-flex items-center gap-4 p-2 bg-gray-100 dark:bg-gray-800 rounded-full">
            <button
              onClick={() => setBillingPeriod('monthly')}
              className={`px-6 py-3 rounded-full font-medium transition-all ${
                billingPeriod === 'monthly'
                  ? 'bg-white dark:bg-gray-700 shadow-md'
                  : 'text-gray-600 dark:text-gray-400'
              }`}
            >
              {language === 'ar' ? 'شهري' : 'Monthly'}
            </button>
            <button
              onClick={() => setBillingPeriod('annual')}
              className={`px-6 py-3 rounded-full font-medium transition-all ${
                billingPeriod === 'annual'
                  ? 'bg-white dark:bg-gray-700 shadow-md'
                  : 'text-gray-600 dark:text-gray-400'
              }`}
            >
              {language === 'ar' ? 'سنوي' : 'Annual'}
              <span className="ml-2 text-xs text-green-500 font-bold">
                {language === 'ar' ? 'وفّر 20%' : 'Save 20%'}
              </span>
            </button>
          </div>
        </motion.div>
      </section>

      {/* Pricing Cards */}
      <section className="container-custom py-16">
        <div className="grid md:grid-cols-3 gap-8">
          {plans.map((plan, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
              className={`glass-card p-8 relative ${
                plan.popular ? 'ring-2 ring-accent-500 scale-105' : ''
              }`}
            >
              {plan.popular && (
                <div className="absolute -top-4 left-1/2 -translate-x-1/2">
                  <div className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-accent-500 to-accent-600 text-white rounded-full text-sm font-medium shadow-glow">
                    <IconSparkles size={16} />
                    {language === 'ar' ? 'الأكثر شعبية' : 'Most Popular'}
                  </div>
                </div>
              )}

              <div className={`inline-flex p-4 rounded-2xl bg-gradient-to-br ${plan.gradient} text-white mb-6`}>
                <IconSparkles size={32} />
              </div>

              <h3 className="text-2xl font-bold mb-2">{plan.name}</h3>
              <p className="text-gray-600 dark:text-gray-400 mb-6">{plan.description}</p>

              <div className="mb-6">
                <span className="text-5xl font-bold gradient-text">
                  ${billingPeriod === 'monthly' ? plan.monthlyPrice : plan.annualPrice}
                </span>
                <span className="text-gray-600 dark:text-gray-400 ml-2">
                  /{billingPeriod === 'monthly'
                    ? (language === 'ar' ? 'شهر' : 'month')
                    : (language === 'ar' ? 'سنة' : 'year')}
                </span>
              </div>

              <ul className="space-y-4 mb-8">
                {plan.features.map((feature, i) => (
                  <li key={i} className="flex items-start gap-3">
                    {feature.included ? (
                      <IconCheck size={20} className="text-green-500 flex-shrink-0 mt-1" />
                    ) : (
                      <IconX size={20} className="text-gray-400 flex-shrink-0 mt-1" />
                    )}
                    <span className={feature.included ? '' : 'text-gray-400 line-through'}>
                      {feature.text}
                    </span>
                  </li>
                ))}
              </ul>

              <Link to="/demo">
                <button
                  className={`w-full py-4 rounded-xl font-semibold transition-all ${
                    plan.popular
                      ? 'magnetic-btn'
                      : 'border-2 border-accent-500 text-accent-500 hover:bg-accent-50 dark:hover:bg-accent-900/20'
                  }`}
                >
                  {language === 'ar' ? 'ابدأ الآن' : 'Get Started'}
                </button>
              </Link>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Features Comparison */}
      <section className="container-custom py-16">
        <motion.h2
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          className="text-4xl font-bold text-center mb-12"
        >
          {language === 'ar' ? 'جميع الخطط تشمل' : 'All Plans Include'}
        </motion.h2>

        <div className="grid md:grid-cols-4 gap-6">
          {[
            { icon: '🔒', text: language === 'ar' ? 'تشفير HIPAA' : 'HIPAA Encryption' },
            { icon: '☁️', text: language === 'ar' ? 'تخزين سحابي' : 'Cloud Storage' },
            { icon: '📱', text: language === 'ar' ? 'تطبيق الهاتف' : 'Mobile App' },
            { icon: '🔄', text: language === 'ar' ? 'تحديثات مجانية' : 'Free Updates' },
          ].map((item, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
              className="glass-card p-6 text-center"
            >
              <div className="text-4xl mb-3">{item.icon}</div>
              <p className="font-medium">{item.text}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* FAQs */}
      <section className="container-custom py-16">
        <motion.h2
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          className="text-4xl font-bold text-center mb-12"
        >
          {language === 'ar' ? 'الأسئلة الشائعة' : 'Frequently Asked Questions'}
        </motion.h2>

        <div className="max-w-3xl mx-auto space-y-4">
          {faqs.map((faq, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
              className="glass-card p-6"
            >
              <h3 className="text-xl font-bold mb-3">{faq.question}</h3>
              <p className="text-gray-600 dark:text-gray-300">{faq.answer}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* CTA */}
      <section className="container-custom py-16">
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
          className="glass-card p-12 text-center"
        >
          <h2 className="text-4xl font-bold mb-6">
            {language === 'ar' ? 'هل لديك أسئلة؟' : 'Have Questions?'}
          </h2>
          <p className="text-xl text-gray-600 dark:text-gray-300 mb-8">
            {language === 'ar'
              ? 'فريقنا هنا للمساعدة. تواصل معنا اليوم!'
              : 'Our team is here to help. Contact us today!'}
          </p>
          <Link to="/demo">
            <button className="magnetic-btn">
              {language === 'ar' ? 'تحدث إلى المبيعات' : 'Talk to Sales'}
            </button>
          </Link>
        </motion.div>
      </section>
    </div>
  );
};

export default Pricing;
