import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useThemeStore } from '@store/themeStore';
import { useAuthStore } from '@store/authStore';
import { IconLoader2, IconAlertCircle, IconMoon, IconSun } from '@tabler/icons-react';
import clsx from 'clsx';
import api from '../utils/api';

const Login = () => {
  const { theme, language, toggleTheme, setLanguage } = useThemeStore();
  const { setAuth } = useAuthStore();
  const navigate = useNavigate();
  
  const [userId, setUserId] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      const data = await api.login(userId, password);
      setAuth(data.access_token, userId, data.roles);
      navigate('/dashboard');
    } catch (err: any) {
      setError(err.message || 'Login failed. Please check your credentials.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={clsx(
      'min-h-screen flex items-center justify-center p-4',
      theme === 'dark' ? 'bg-gray-900' : 'bg-gray-50'
    )}>
      {/* Theme & Language Toggle */}
      <div className="fixed top-4 right-4 flex gap-2">
        <button
          onClick={toggleTheme}
          className={clsx(
            'p-2 rounded-lg transition-colors',
            theme === 'dark' 
              ? 'bg-gray-800 hover:bg-gray-700 text-white' 
              : 'bg-white hover:bg-gray-100 text-gray-900'
          )}
        >
          {theme === 'dark' ? <IconSun size={20} /> : <IconMoon size={20} />}
        </button>
        <button
          onClick={() => setLanguage(language === 'ar' ? 'en' : 'ar')}
          className={clsx(
            'px-3 py-2 rounded-lg transition-colors text-sm font-medium',
            theme === 'dark' 
              ? 'bg-gray-800 hover:bg-gray-700 text-white' 
              : 'bg-white hover:bg-gray-100 text-gray-900'
          )}
        >
          {language === 'ar' ? 'EN' : 'عربي'}
        </button>
      </div>

      <div className={clsx(
        'w-full max-w-md rounded-2xl p-8 border',
        theme === 'dark' 
          ? 'bg-gray-800 border-gray-700' 
          : 'bg-white border-gray-200'
      )}>
        {/* Logo */}
        <div className="text-center mb-8">
          <div className="w-16 h-16 rounded-2xl bg-blue-600 flex items-center justify-center mx-auto mb-4">
            <span className="text-3xl font-bold text-white">H</span>
          </div>
          <h1 className={clsx(
            'text-2xl font-bold',
            theme === 'dark' ? 'text-white' : 'text-gray-900'
          )}>
            HealthTech AI
          </h1>
          <p className={clsx(
            'mt-1',
            theme === 'dark' ? 'text-gray-400' : 'text-gray-500'
          )}>
            {language === 'ar' ? 'نظام التوثيق الطبي الذكي' : 'Smart Medical Documentation System'}
          </p>
        </div>

        {/* Login Form */}
        <form onSubmit={handleSubmit} className="space-y-4">
          {error && (
            <div className="flex items-center gap-2 p-3 rounded-lg bg-red-500/10 text-red-500 text-sm">
              <IconAlertCircle size={18} />
              <span>{error}</span>
            </div>
          )}

          <div>
            <label className={clsx(
              'block text-sm font-medium mb-2',
              theme === 'dark' ? 'text-gray-300' : 'text-gray-700'
            )}>
              {language === 'ar' ? 'معرف المستخدم' : 'User ID'}
            </label>
            <input
              type="text"
              value={userId}
              onChange={(e) => setUserId(e.target.value)}
              className={clsx(
                'w-full px-4 py-3 rounded-xl border transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500',
                theme === 'dark'
                  ? 'bg-gray-700 border-gray-600 text-white placeholder-gray-400'
                  : 'bg-gray-50 border-gray-200 text-gray-900 placeholder-gray-400'
              )}
              placeholder={language === 'ar' ? 'أدخل معرف المستخدم' : 'Enter your user ID'}
              required
            />
          </div>

          <div>
            <label className={clsx(
              'block text-sm font-medium mb-2',
              theme === 'dark' ? 'text-gray-300' : 'text-gray-700'
            )}>
              {language === 'ar' ? 'كلمة المرور' : 'Password'}
            </label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className={clsx(
                'w-full px-4 py-3 rounded-xl border transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500',
                theme === 'dark'
                  ? 'bg-gray-700 border-gray-600 text-white placeholder-gray-400'
                  : 'bg-gray-50 border-gray-200 text-gray-900 placeholder-gray-400'
              )}
              placeholder={language === 'ar' ? 'أدخل كلمة المرور' : 'Enter your password'}
              required
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className={clsx(
              'w-full py-3 rounded-xl font-medium transition-colors flex items-center justify-center gap-2',
              loading
                ? 'bg-blue-600/50 cursor-not-allowed'
                : 'bg-blue-600 hover:bg-blue-700',
              'text-white'
            )}
          >
            {loading && <IconLoader2 size={20} className="animate-spin" />}
            {language === 'ar' ? 'تسجيل الدخول' : 'Sign In'}
          </button>

          {/* OIDC SSO — shown only when the IdP is configured */}
          {import.meta.env.VITE_OIDC_ENABLED === 'true' && (
            <>
              <div className="relative flex items-center gap-2">
                <div className="flex-grow border-t border-gray-300 dark:border-gray-600" />
                <span className={clsx('text-xs', theme === 'dark' ? 'text-gray-400' : 'text-gray-500')}>
                  {language === 'ar' ? 'أو' : 'or'}
                </span>
                <div className="flex-grow border-t border-gray-300 dark:border-gray-600" />
              </div>
              <a
                href="/auth/oidc/login"
                className={clsx(
                  'w-full py-3 rounded-xl font-medium transition-colors flex items-center justify-center gap-2 border',
                  theme === 'dark'
                    ? 'border-gray-600 text-gray-200 hover:bg-gray-700'
                    : 'border-gray-300 text-gray-700 hover:bg-gray-50'
                )}
              >
                {language === 'ar' ? 'الدخول عبر المؤسسة (SSO)' : 'Sign in with SSO'}
              </a>
            </>
          )}
        </form>

        {/* Demo Credentials Hint — development only */}
        {import.meta.env.DEV && (
          <div className={clsx(
            'mt-6 p-4 rounded-xl text-sm',
            theme === 'dark' ? 'bg-gray-700/50 text-gray-400' : 'bg-gray-50 text-gray-500'
          )}>
            <p className="font-medium mb-2">
              {language === 'ar' ? 'بيانات تجريبية:' : 'Demo credentials:'}
            </p>
            <p>User ID: <code className="px-1.5 py-0.5 rounded bg-black/10 dark:bg-white/10">demo@healthtech.com</code></p>
            <p>Password: <code className="px-1.5 py-0.5 rounded bg-black/10 dark:bg-white/10">demo123</code></p>
          </div>
        )}
      </div>
    </div>
  );
};

export default Login;
