import { motion } from 'framer-motion';
import { useThemeStore } from '@store/themeStore';
import {
  IconTarget,
  IconHeart,
  IconUsers,
  IconAward,
  IconTrendingUp,
  IconShieldCheck
} from '@tabler/icons-react';

const About = () => {
  const { language } = useThemeStore();

  const stats = [
    { value: '98%', label: language === 'ar' ? 'دقة' : 'Accuracy' },
    { value: '50K+', label: language === 'ar' ? 'مرضى' : 'Patients' },
    { value: '1,000+', label: language === 'ar' ? 'أطباء' : 'Doctors' },
    { value: '24/7', label: language === 'ar' ? 'الدعم' : 'Support' },
  ];

  const values = [
    {
      icon: <IconHeart size={32} />,
      title: language === 'ar' ? 'رعاية المرضى' : 'Patient Care First',
      description: language === 'ar'
        ? 'نضع رعاية المرضى في المقام الأول في كل ما نفعله'
        : 'We put patient care at the heart of everything we do',
    },
    {
      icon: <IconShieldCheck size={32} />,
      title: language === 'ar' ? 'الأمان والخصوصية' : 'Security & Privacy',
      description: language === 'ar'
        ? 'حماية البيانات الطبية بأعلى معايير الأمان'
        : 'Protecting medical data with the highest security standards',
    },
    {
      icon: <IconTrendingUp size={32} />,
      title: language === 'ar' ? 'الابتكار المستمر' : 'Continuous Innovation',
      description: language === 'ar'
        ? 'نستخدم أحدث تقنيات الذكاء الاصطناعي لتحسين الرعاية الصحية'
        : 'Using latest AI technology to improve healthcare delivery',
    },
  ];

  const team = [
    {
      name: 'Dr. Sarah Johnson',
      role: language === 'ar' ? 'المدير الطبي' : 'Chief Medical Officer',
      image: '👩‍⚕️',
    },
    {
      name: 'Michael Chen',
      role: language === 'ar' ? 'مدير التكنولوجيا' : 'Chief Technology Officer',
      image: '👨‍💻',
    },
    {
      name: 'Dr. Ahmed Hassan',
      role: language === 'ar' ? 'مدير الأبحاث' : 'Head of Research',
      image: '👨‍🔬',
    },
    {
      name: 'Lisa Williams',
      role: language === 'ar' ? 'مديرة المنتج' : 'Product Manager',
      image: '👩‍💼',
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
              {language === 'ar' ? 'من نحن' : 'About Us'}
            </span>
          </h1>
          <p className="text-xl md:text-2xl text-gray-600 dark:text-gray-300">
            {language === 'ar'
              ? 'نحن نعيد تعريف الرعاية الصحية من خلال الذكاء الاصطناعي'
              : "We're redefining healthcare through artificial intelligence"}
          </p>
        </motion.div>
      </section>

      {/* Stats */}
      <section className="container-custom py-16">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
          {stats.map((stat, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
              className="glass-card p-8 text-center"
            >
              <div className="text-4xl md:text-5xl font-bold gradient-text mb-2">
                {stat.value}
              </div>
              <div className="text-gray-600 dark:text-gray-300">{stat.label}</div>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Mission */}
      <section className="container-custom py-16">
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          className="glass-card p-12"
        >
          <div className="flex items-center gap-4 mb-6">
            <IconTarget size={40} className="text-accent-500" />
            <h2 className="text-3xl font-bold">
              {language === 'ar' ? 'مهمتنا' : 'Our Mission'}
            </h2>
          </div>
          <p className="text-xl text-gray-600 dark:text-gray-300 leading-relaxed mb-6">
            {language === 'ar'
              ? 'مهمتنا هي تمكين مقدمي الرعاية الصحية من خلال أحدث تقنيات الذكاء الاصطناعي لتحسين رعاية المرضى، وتقليل العبء الإداري، وتعزيز نتائج العلاج. نحن نؤمن بأن التكنولوجيا يجب أن تعمل من أجل الأطباء، وليس العكس.'
              : 'Our mission is to empower healthcare providers with cutting-edge AI technology to improve patient care, reduce administrative burden, and enhance treatment outcomes. We believe technology should work for doctors, not the other way around.'}
          </p>
          <p className="text-lg text-gray-600 dark:text-gray-300 leading-relaxed">
            {language === 'ar'
              ? 'من خلال الجمع بين خبرة الأطباء وقوة الذكاء الاصطناعي، نقوم بإنشاء أدوات تساعد المهنيين الطبيين على التركيز على ما يهم حقًا: مرضاهم.'
              : 'By combining medical expertise with the power of AI, we create tools that help medical professionals focus on what truly matters: their patients.'}
          </p>
        </motion.div>
      </section>

      {/* Values */}
      <section className="container-custom py-16">
        <motion.h2
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          className="text-4xl font-bold text-center mb-12"
        >
          {language === 'ar' ? 'قيمنا' : 'Our Values'}
        </motion.h2>

        <div className="grid md:grid-cols-3 gap-8">
          {values.map((value, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
              className="glass-card p-8"
            >
              <div className="inline-flex p-4 rounded-full bg-accent-100 dark:bg-accent-900/20 text-accent-500 mb-6">
                {value.icon}
              </div>
              <h3 className="text-2xl font-bold mb-4">{value.title}</h3>
              <p className="text-gray-600 dark:text-gray-300">{value.description}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Team */}
      <section className="container-custom py-16">
        <motion.h2
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          className="text-4xl font-bold text-center mb-12"
        >
          {language === 'ar' ? 'فريقنا' : 'Our Team'}
        </motion.h2>

        <div className="grid md:grid-cols-4 gap-8">
          {team.map((member, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
              className="glass-card p-6 text-center group hover:scale-105 transition-transform"
            >
              <div className="text-6xl mb-4">{member.image}</div>
              <h3 className="text-xl font-bold mb-2">{member.name}</h3>
              <p className="text-gray-600 dark:text-gray-400">{member.role}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Certifications */}
      <section className="container-custom py-16">
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
          className="glass-card p-12 text-center"
        >
          <IconAward size={48} className="mx-auto text-accent-500 mb-6" />
          <h2 className="text-3xl font-bold mb-4">
            {language === 'ar' ? 'الشهادات والامتثال' : 'Certifications & Compliance'}
          </h2>
          <p className="text-xl text-gray-600 dark:text-gray-300 mb-8">
            {language === 'ar'
              ? 'نحن متوافقون مع HIPAA، ISO 27001، وSOC 2 Type II'
              : 'HIPAA Compliant, ISO 27001, and SOC 2 Type II Certified'}
          </p>
          <div className="flex justify-center gap-8 flex-wrap">
            <div className="px-6 py-3 bg-accent-100 dark:bg-accent-900/20 rounded-full font-medium">
              HIPAA Compliant
            </div>
            <div className="px-6 py-3 bg-accent-100 dark:bg-accent-900/20 rounded-full font-medium">
              ISO 27001
            </div>
            <div className="px-6 py-3 bg-accent-100 dark:bg-accent-900/20 rounded-full font-medium">
              SOC 2 Type II
            </div>
          </div>
        </motion.div>
      </section>
    </div>
  );
};

export default About;
