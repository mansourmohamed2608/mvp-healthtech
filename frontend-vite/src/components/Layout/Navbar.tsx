import { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Menu, X, Moon, Sun, Globe } from 'lucide-react';
import { useThemeStore } from '@store/themeStore';
import { useScrollPosition } from '@hooks/useScrollPosition';
import MagneticButton from '@components/UI/MagneticButton';
import clsx from 'clsx';

const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false);
  const { theme, language, toggleTheme, setLanguage } = useThemeStore();
  const { scrollY, scrollDirection } = useScrollPosition();
  const location = useLocation();

  const isScrolled = scrollY > 50;
  const shouldHide = scrollDirection === 'down' && scrollY > 200;

  const navItems = [
    { name: language === 'ar' ? 'الرئيسية' : 'Home', path: '/' },
    { name: language === 'ar' ? 'الميزات' : 'Features', path: '/features' },
    { name: language === 'ar' ? 'المساعد الصوتي' : 'Voice Agent', path: '/voice-agent' },
    { name: language === 'ar' ? 'الملاحظات السريرية' : 'Clinical Notes', path: '/features/clinical-notes' },
    { name: language === 'ar' ? 'لوحة التحكم' : 'Dashboard', path: '/dashboard' },
    { name: language === 'ar' ? 'من نحن' : 'About', path: '/about' },
    { name: language === 'ar' ? 'الأسعار' : 'Pricing', path: '/pricing' },
  ];

  return (
    <motion.nav
      initial={{ y: 0 }}
      animate={{ y: shouldHide ? -100 : 0 }}
      transition={{ duration: 0.3, ease: 'easeInOut' }}
      className={clsx(
        'fixed top-0 left-0 right-0 z-50 transition-all duration-300',
        isScrolled
          ? 'glass backdrop-blur-2xl border-b border-white/10'
          : 'bg-transparent'
      )}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-20">
          {/* Logo */}
          <Link to="/" className="flex items-center space-x-3 group">
            <motion.div
              whileHover={{ scale: 1.05, rotate: 5 }}
              whileTap={{ scale: 0.95 }}
              className="relative"
            >
              <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-accent-500 to-accent-600 flex items-center justify-center shadow-glow">
                <span className="text-2xl font-bold text-white">H</span>
              </div>
              <div className="absolute inset-0 rounded-xl bg-gradient-to-br from-accent-400 to-accent-500 blur-xl opacity-50 group-hover:opacity-75 transition-opacity" />
            </motion.div>
            <div>
              <h1 className="text-xl font-display font-bold bg-gradient-to-r from-accent-500 to-accent-600 bg-clip-text text-transparent">
                {language === 'ar' ? 'هيلث تك' : 'HealthTech AI'}
              </h1>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                {language === 'ar' ? 'توثيق طبي ذكي' : 'Smart Medical Documentation'}
              </p>
            </div>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden lg:flex items-center space-x-8">
            {navItems.map((item) => (
              <Link
                key={item.path}
                to={item.path}
                className={clsx(
                  'relative px-4 py-2 text-sm font-medium transition-colors duration-200',
                  location.pathname === item.path
                    ? 'text-accent-500'
                    : 'text-gray-700 dark:text-gray-300 hover:text-accent-500'
                )}
              >
                {item.name}
                {location.pathname === item.path && (
                  <motion.div
                    layoutId="navbar-indicator"
                    className="absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r from-accent-500 to-accent-600"
                    transition={{ type: 'spring', stiffness: 380, damping: 30 }}
                  />
                )}
              </Link>
            ))}
          </div>

          {/* Actions */}
          <div className="flex items-center space-x-4">
            {/* Language Toggle */}
            <button
              onClick={() => setLanguage(language === 'ar' ? 'en' : 'ar')}
              className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-dark-800 transition-colors"
              aria-label="Toggle Language"
            >
              <Globe className="w-5 h-5" />
            </button>

            {/* Theme Toggle */}
            <motion.button
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              onClick={toggleTheme}
              className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-dark-800 transition-colors"
              aria-label="Toggle Theme"
            >
              <AnimatePresence mode="wait">
                {theme === 'dark' ? (
                  <motion.div
                    key="sun"
                    initial={{ rotate: -90, opacity: 0 }}
                    animate={{ rotate: 0, opacity: 1 }}
                    exit={{ rotate: 90, opacity: 0 }}
                    transition={{ duration: 0.2 }}
                  >
                    <Sun className="w-5 h-5 text-yellow-500" />
                  </motion.div>
                ) : (
                  <motion.div
                    key="moon"
                    initial={{ rotate: 90, opacity: 0 }}
                    animate={{ rotate: 0, opacity: 1 }}
                    exit={{ rotate: -90, opacity: 0 }}
                    transition={{ duration: 0.2 }}
                  >
                    <Moon className="w-5 h-5 text-accent-500" />
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.button>

            {/* CTA Button */}
            <div className="hidden lg:block">
              <Link to="/demo">
                <MagneticButton>
                  {language === 'ar' ? 'جرب المنصة' : 'Try Demo'}
                </MagneticButton>
              </Link>
            </div>

            {/* Mobile Menu Button */}
            <button
              onClick={() => setIsOpen(!isOpen)}
              className="lg:hidden p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-dark-800 transition-colors"
              aria-label="Toggle Menu"
            >
              {isOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Menu */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3 }}
            className="lg:hidden glass backdrop-blur-2xl border-t border-white/10"
          >
            <div className="px-4 py-6 space-y-3">
              {navItems.map((item) => (
                <Link
                  key={item.path}
                  to={item.path}
                  onClick={() => setIsOpen(false)}
                  className={clsx(
                    'block px-4 py-3 rounded-lg text-base font-medium transition-colors',
                    location.pathname === item.path
                      ? 'bg-accent-500/10 text-accent-500'
                      : 'hover:bg-gray-100 dark:hover:bg-dark-800'
                  )}
                >
                  {item.name}
                </Link>
              ))}
              <Link to="/demo" onClick={() => setIsOpen(false)}>
                <button className="w-full magnetic-btn">
                  {language === 'ar' ? 'جرب المنصة' : 'Try Demo'}
                </button>
              </Link>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.nav>
  );
};

export default Navbar;
