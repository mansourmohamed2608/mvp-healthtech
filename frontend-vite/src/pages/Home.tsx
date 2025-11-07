import { useRef } from 'react';
import { motion, useScroll, useTransform } from 'framer-motion';
import { Link } from 'react-router-dom';
import {
  Sparkles,
  Zap,
  Shield,
  TrendingUp,
  ArrowRight,
  Mic,
  FileText,
  Database,
  Stethoscope,
  CheckCircle2,
  Star
} from 'lucide-react';
import { useThemeStore } from '@store/themeStore';
import MagneticButton from '@components/UI/MagneticButton';
import Hero3D from '@components/3D/Hero3D';

const Home = () => {
  const { language } = useThemeStore();
  const containerRef = useRef(null);
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ['start start', 'end start'],
  });

  const opacity = useTransform(scrollYProgress, [0, 0.5], [1, 0]);
  const scale = useTransform(scrollYProgress, [0, 0.5], [1, 0.8]);

  const features = [
    {
      icon: Mic,
      title: language === 'ar' ? 'النسخ الصوتي بالذكاء الاصطناعي' : 'AI Voice Transcription',
      description: language === 'ar'
        ? 'تحويل المحادثات الطبية إلى نصوص دقيقة بنسبة 99% مع دعم اللهجات المحلية'
        : 'Convert medical conversations to accurate text with 99% accuracy and local dialect support',
      gradient: 'from-blue-500 to-cyan-500',
      delay: 0.1,
    },
    {
      icon: FileText,
      title: language === 'ar' ? 'توليد النصوص الطبية' : 'Clinical Notes Generation',
      description: language === 'ar'
        ? 'إنشاء سجلات طبية منظمة ومتوافقة مع المعايير الطبية تلقائياً'
        : 'Automatically generate structured clinical notes compliant with medical standards',
      gradient: 'from-purple-500 to-pink-500',
      delay: 0.2,
    },
    {
      icon: Database,
      title: language === 'ar' ? 'تكامل FHIR' : 'FHIR Integration',
      description: language === 'ar'
        ? 'تكامل سلس مع الأنظمة الصحية عبر معيار FHIR العالمي'
        : 'Seamless integration with health systems through FHIR global standard',
      gradient: 'from-green-500 to-emerald-500',
      delay: 0.3,
    },
    {
      icon: Stethoscope,
      title: language === 'ar' ? 'توليد SOAP' : 'SOAP Generation',
      description: language === 'ar'
        ? 'إنشاء ملاحظات SOAP كاملة من المحادثات الطبية بشكل فوري'
        : 'Generate complete SOAP notes from medical conversations instantly',
      gradient: 'from-orange-500 to-red-500',
      delay: 0.4,
    },
  ];

  const stats = [
    { value: '99%', label: language === 'ar' ? 'دقة النسخ' : 'Transcription Accuracy' },
    { value: '10x', label: language === 'ar' ? 'أسرع في التوثيق' : 'Faster Documentation' },
    { value: '50K+', label: language === 'ar' ? 'مستند معالج' : 'Documents Processed' },
    { value: '24/7', label: language === 'ar' ? 'دعم متواصل' : 'Support Available' },
  ];

  const benefits = [
    { text: language === 'ar' ? 'تقليل وقت التوثيق بنسبة 80%' : 'Reduce documentation time by 80%' },
    { text: language === 'ar' ? 'تحسين دقة السجلات الطبية' : 'Improve medical record accuracy' },
    { text: language === 'ar' ? 'تكامل سلس مع الأنظمة الحالية' : 'Seamless integration with existing systems' },
    { text: language === 'ar' ? 'امتثال كامل للمعايير الصحية' : 'Full compliance with health standards' },
    { text: language === 'ar' ? 'تشفير متقدم للبيانات' : 'Advanced data encryption' },
    { text: language === 'ar' ? 'دعم لغات ولهجات متعددة' : 'Multi-language and dialect support' },
  ];

  const testimonials = [
    {
      name: 'Dr. Sarah المحمد',
      role: language === 'ar' ? 'طبيبة عامة' : 'General Practitioner',
      content: language === 'ar'
        ? 'غيرت هذه المنصة طريقة عملي بالكامل. الآن أركز على المريض بدلاً من الكتابة.'
        : 'This platform completely transformed my workflow. Now I focus on patients instead of paperwork.',
      rating: 5,
      image: '👩‍⚕️',
    },
    {
      name: 'Dr. Ahmed السعيد',
      role: language === 'ar' ? 'أخصائي قلب' : 'Cardiologist',
      content: language === 'ar'
        ? 'الدقة مذهلة وتكامل FHIR سلس. أوصي بها لجميع الزملاء.'
        : 'The accuracy is amazing and FHIR integration is seamless. I recommend it to all colleagues.',
      rating: 5,
      image: '👨‍⚕️',
    },
    {
      name: 'Dr. Fatima خان',
      role: language === 'ar' ? 'أخصائية أطفال' : 'Pediatrician',
      content: language === 'ar'
        ? 'وفرت ساعات من وقتي كل يوم. أفضل استثمار للعيادة.'
        : 'Saved me hours every day. Best investment for my clinic.',
      rating: 5,
      image: '👩‍⚕️',
    },
  ];

  return (
    <div ref={containerRef} className="relative">
      {/* Hero Section with 3D */}
      <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
        {/* 3D Background */}
        <div className="absolute inset-0 z-0">
          <Hero3D />
        </div>

        <motion.div
          style={{ opacity, scale }}
          className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-32 text-center"
        >
          {/* Badge */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="inline-flex items-center space-x-2 px-4 py-2 rounded-full glass mb-8"
          >
            <Sparkles className="w-4 h-4 text-accent-500" />
            <span className="text-sm font-medium">
              {language === 'ar' ? '🎉 إطلاق المنصة الجديدة' : '🎉 New Platform Launch'}
            </span>
          </motion.div>

          {/* Main Headline with Kinetic Typography */}
          <motion.h1
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="text-5xl md:text-7xl lg:text-8xl font-display font-bold mb-6 leading-tight"
          >
            <span className="block shimmer-text">
              {language === 'ar' ? 'توثيق طبي' : 'Medical Documentation'}
            </span>
            <span className="block glow-text mt-2">
              {language === 'ar' ? 'بقوة الذكاء الاصطناعي' : 'Powered by AI'}
            </span>
          </motion.h1>

          {/* Subtitle */}
          <motion.p
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            className="text-xl md:text-2xl text-gray-600 dark:text-gray-300 mb-12 max-w-3xl mx-auto"
          >
            {language === 'ar'
              ? 'حول المحادثات الطبية إلى سجلات منظمة بدقة فائقة وسرعة مذهلة'
              : 'Transform medical conversations into structured records with exceptional accuracy and speed'}
          </motion.p>

          {/* CTA Buttons */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.6 }}
            className="flex flex-col sm:flex-row items-center justify-center gap-4"
          >
            <Link to="/demo">
              <MagneticButton className="group">
                <span className="flex items-center space-x-2">
                  <Zap className="w-5 h-5" />
                  <span>{language === 'ar' ? 'جرب المنصة مجاناً' : 'Try Free Demo'}</span>
                  <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                </span>
              </MagneticButton>
            </Link>
            <Link to="/features">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="px-8 py-4 rounded-full font-semibold glass backdrop-blur-xl border border-white/20 hover:border-accent-500/50 transition-all"
              >
                {language === 'ar' ? 'استكشف الميزات' : 'Explore Features'}
              </motion.button>
            </Link>
          </motion.div>

          {/* Trust Badges */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 1, delay: 0.8 }}
            className="mt-16 flex flex-wrap items-center justify-center gap-8 text-sm text-gray-500 dark:text-gray-400"
          >
            <div className="flex items-center space-x-2">
              <Shield className="w-5 h-5 text-green-500" />
              <span>{language === 'ar' ? 'معتمد من وزارة الصحة' : 'MOH Certified'}</span>
            </div>
            <div className="flex items-center space-x-2">
              <CheckCircle2 className="w-5 h-5 text-blue-500" />
              <span>{language === 'ar' ? 'متوافق مع HIPAA' : 'HIPAA Compliant'}</span>
            </div>
            <div className="flex items-center space-x-2">
              <Star className="w-5 h-5 text-yellow-500" />
              <span>{language === 'ar' ? 'تقييم 4.9/5 من الأطباء' : '4.9/5 from Doctors'}</span>
            </div>
          </motion.div>
        </motion.div>

        {/* Scroll Indicator */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1, delay: 1 }}
          className="absolute bottom-8 left-1/2 transform -translate-x-1/2"
        >
          <motion.div
            animate={{ y: [0, 10, 0] }}
            transition={{ duration: 2, repeat: Infinity }}
            className="w-6 h-10 rounded-full border-2 border-gray-400 dark:border-gray-600 flex items-start justify-center p-2"
          >
            <motion.div className="w-1 h-2 bg-gray-400 dark:bg-gray-600 rounded-full" />
          </motion.div>
        </motion.div>
      </section>

      {/* Stats Section */}
      <section className="relative py-20 overflow-hidden">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {stats.map((stat, index) => (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                className="text-center"
              >
                <div className="text-4xl md:text-5xl font-bold glow-text mb-2">
                  {stat.value}
                </div>
                <div className="text-gray-600 dark:text-gray-400">
                  {stat.label}
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section - Bento Grid */}
      <section className="relative py-32">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl md:text-5xl font-display font-bold mb-4">
              <span className="shimmer-text">
                {language === 'ar' ? 'ميزات قوية' : 'Powerful Features'}
              </span>
            </h2>
            <p className="text-xl text-gray-600 dark:text-gray-300 max-w-2xl mx-auto">
              {language === 'ar'
                ? 'كل ما تحتاجه لتحويل الرعاية الصحية'
                : 'Everything you need to transform healthcare'}
            </p>
          </motion.div>

          <div className="bento-grid">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: feature.delay }}
                className={`bento-item card-3d group ${index === 0 ? 'bento-lg' : index === features.length - 1 ? 'bento-md' : 'bento-sm'}`}
              >
                <div className="relative h-full flex flex-col">
                  <div className={`w-16 h-16 rounded-2xl bg-gradient-to-br ${feature.gradient} flex items-center justify-center mb-6 group-hover:scale-110 transition-transform`}>
                    <feature.icon className="w-8 h-8 text-white" />
                  </div>

                  <h3 className="text-2xl font-bold mb-3">{feature.title}</h3>
                  <p className="text-gray-600 dark:text-gray-400 flex-1">
                    {feature.description}
                  </p>

                  <Link
                    to={`/features/${feature.title.toLowerCase().replace(/\s+/g, '-')}`}
                    className="mt-6 inline-flex items-center text-accent-500 hover:text-accent-600 font-medium group-hover:translate-x-2 transition-transform"
                  >
                    {language === 'ar' ? 'اعرف المزيد' : 'Learn More'}
                    <ArrowRight className="w-4 h-4 ml-2" />
                  </Link>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Benefits Section */}
      <section className="relative py-32 glass">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid lg:grid-cols-2 gap-16 items-center">
            <motion.div
              initial={{ opacity: 0, x: -50 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.8 }}
            >
              <h2 className="text-4xl md:text-5xl font-display font-bold mb-6">
                {language === 'ar' ? 'لماذا تختار هيلث تك؟' : 'Why Choose HealthTech?'}
              </h2>
              <p className="text-xl text-gray-600 dark:text-gray-300 mb-8">
                {language === 'ar'
                  ? 'نوفر لك الوقت والجهد مع ضمان أعلى معايير الجودة والدقة'
                  : 'We save you time and effort while ensuring the highest standards of quality and accuracy'}
              </p>

              <div className="space-y-4">
                {benefits.map((benefit, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, x: -20 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.5, delay: index * 0.1 }}
                    className="flex items-start space-x-3 group"
                  >
                    <CheckCircle2 className="w-6 h-6 text-green-500 flex-shrink-0 mt-1 group-hover:scale-110 transition-transform" />
                    <span className="text-lg">{benefit.text}</span>
                  </motion.div>
                ))}
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 50 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.8 }}
              className="relative"
            >
              <div className="aspect-square rounded-3xl glass p-8 relative overflow-hidden">
                <div className="absolute inset-0 bg-gradient-to-br from-accent-500/20 to-accent-600/20" />
                <div className="relative z-10 h-full flex items-center justify-center text-9xl">
                  🏥
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Testimonials */}
      <section className="relative py-32">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl md:text-5xl font-display font-bold mb-4">
              {language === 'ar' ? 'ماذا يقول الأطباء' : 'What Doctors Say'}
            </h2>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-8">
            {testimonials.map((testimonial, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                className="glass-card hover:shadow-glow"
              >
                <div className="flex items-center mb-4">
                  {[...Array(testimonial.rating)].map((_, i) => (
                    <Star key={i} className="w-5 h-5 text-yellow-500 fill-yellow-500" />
                  ))}
                </div>
                <p className="text-gray-600 dark:text-gray-300 mb-6 italic">
                  "{testimonial.content}"
                </p>
                <div className="flex items-center space-x-3">
                  <div className="text-4xl">{testimonial.image}</div>
                  <div>
                    <div className="font-semibold">{testimonial.name}</div>
                    <div className="text-sm text-gray-500">{testimonial.role}</div>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="relative py-32">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
            className="glass-card p-12"
          >
            <TrendingUp className="w-16 h-16 mx-auto mb-6 text-accent-500" />
            <h2 className="text-4xl md:text-5xl font-display font-bold mb-6">
              {language === 'ar' ? 'جاهز للبدء؟' : 'Ready to Get Started?'}
            </h2>
            <p className="text-xl text-gray-600 dark:text-gray-300 mb-8">
              {language === 'ar'
                ? 'انضم إلى آلاف الأطباء الذين يستخدمون هيلث تك يومياً'
                : 'Join thousands of doctors using HealthTech daily'}
            </p>
            <Link to="/demo">
              <MagneticButton>
                {language === 'ar' ? 'ابدأ التجربة المجانية' : 'Start Free Trial'}
              </MagneticButton>
            </Link>
          </motion.div>
        </div>
      </section>
    </div>
  );
};

export default Home;
