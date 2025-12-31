import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  Github,
  Twitter,
  Linkedin,
  Mail,
  Heart,
} from 'lucide-react';
import { useThemeStore } from '@store/themeStore';

const Footer = () => {
  const { language } = useThemeStore();
  const currentYear = new Date().getFullYear();

  const footerLinks = {
    product: [
      { name: language === 'ar' ? 'المساعد الصوتي' : 'Voice Agent', path: '/voice-agent' },
      { name: language === 'ar' ? 'الملاحظات السريرية' : 'Clinical Notes', path: '/clinical-notes' },
    ],
    features: [],
    company: [],
  };

  const socialLinks = [
    { icon: Github, href: 'https://github.com', label: 'GitHub' },
    { icon: Twitter, href: 'https://twitter.com', label: 'Twitter' },
    { icon: Linkedin, href: 'https://linkedin.com', label: 'LinkedIn' },
    { icon: Mail, href: 'mailto:info@healthtech.ai', label: 'Email' },
  ];

  return (
    <footer className="relative mt-32 border-t border-gray-200 dark:border-dark-800 grain-texture">
      {/* Gradient Overlay */}
      <div className="absolute inset-0 bg-gradient-to-b from-transparent via-accent-500/5 to-accent-600/10 pointer-events-none" />

      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-12 mb-12">
          {/* Brand Section */}
          <div className="lg:col-span-2">
            <div className="flex items-center space-x-3 mb-6">
              <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-accent-500 to-accent-600 flex items-center justify-center shadow-glow">
                <span className="text-2xl font-bold text-white">H</span>
              </div>
              <div>
                <h3 className="text-xl font-display font-bold">
                  {language === 'ar' ? 'هيلث تك' : 'HealthTech AI'}
                </h3>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  {language === 'ar' ? 'توثيق طبي ذكي' : 'Smart Medical Documentation'}
                </p>
              </div>
            </div>
            <p className="text-gray-600 dark:text-gray-400 mb-6 max-w-md">
              {language === 'ar'
                ? 'تحويل الرعاية الصحية من خلال توثيق طبي مدعوم بالذكاء الاصطناعي، نسخ صوتي، وتكامل سلس مع FHIR.'
                : 'Transforming healthcare documentation with AI-powered medical transcription, voice recognition, and seamless FHIR integration.'}
            </p>

            {/* Social Links */}
            <div className="flex space-x-4">
              {socialLinks.map((social) => (
                <motion.a
                  key={social.label}
                  href={social.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  whileHover={{ scale: 1.1, y: -2 }}
                  whileTap={{ scale: 0.95 }}
                  className="p-2 rounded-lg bg-gray-100 dark:bg-dark-800 hover:bg-accent-500/10 hover:text-accent-500 transition-colors"
                  aria-label={social.label}
                >
                  <social.icon className="w-5 h-5" />
                </motion.a>
              ))}
            </div>
          </div>

          {/* Product Links */}
          <div>
            <h4 className="font-semibold mb-4 text-gray-900 dark:text-white">
              {language === 'ar' ? 'المنتج' : 'Product'}
            </h4>
            <ul className="space-y-3">
              {footerLinks.product.map((link) => (
                <li key={link.path}>
                  <Link
                    to={link.path}
                    className="text-gray-600 dark:text-gray-400 hover:text-accent-500 transition-colors"
                  >
                    {link.name}
                  </Link>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="pt-8 border-t border-gray-200 dark:border-dark-800">
          <div className="flex flex-col md:flex-row justify-between items-center space-y-4 md:space-y-0">
            <p className="text-sm text-gray-600 dark:text-gray-400 flex items-center">
              {language === 'ar' ? 'صنع بكل' : 'Made with'}{' '}
              <Heart className="w-4 h-4 mx-1 text-red-500 animate-pulse" />{' '}
              {language === 'ar' ? 'للرعاية الصحية' : 'for Healthcare'}
            </p>

            <p className="text-sm text-gray-600 dark:text-gray-400">
              © {currentYear} {language === 'ar' ? 'هيلث تك.' : 'HealthTech AI.'}{' '}
              {language === 'ar' ? 'جميع الحقوق محفوظة.' : 'All rights reserved.'}
            </p>

          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
