import { useState } from 'react';
import { Outlet, NavLink, useNavigate } from 'react-router-dom';
import { useThemeStore } from '@store/themeStore';
import { useAuthStore } from '@store/authStore';
import {
  IconPhone,
  IconFileText,
  IconDashboard,
  IconLogout,
  IconMenu2,
  IconX,
  IconMoon,
  IconSun,
  IconUser,
  IconChevronLeft,
  IconChevronRight,
  IconBrain
} from '@tabler/icons-react';
import clsx from 'clsx';

const ClinicalLayout = () => {
  const { theme, language, toggleTheme, setLanguage } = useThemeStore();
  const { token, userId, clearAuth } = useAuthStore();
  const navigate = useNavigate();
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const handleLogout = () => {
    clearAuth();
    navigate('/login');
  };

  const navItems = [
    {
      name: language === 'ar' ? 'لوحة التحكم' : 'Dashboard',
      path: '/dashboard',
      icon: IconDashboard
    },
    {
      name: language === 'ar' ? 'المساعد الصوتي' : 'Voice Agent',
      path: '/voice-agent',
      icon: IconPhone
    },
    {
      name: language === 'ar' ? 'الملاحظات السريرية' : 'Clinical Notes',
      path: '/clinical-notes',
      icon: IconFileText
    },
    {
      name: language === 'ar' ? 'قاعدة المعرفة' : 'Knowledge Base',
      path: '/knowledge-base',
      icon: IconBrain
    }
  ];

  return (
    <div className={clsx(
      'min-h-screen flex',
      theme === 'dark' ? 'bg-gray-900 text-white' : 'bg-gray-50 text-gray-900'
    )}>
      {/* Sidebar - Desktop */}
      <aside className={clsx(
        'hidden lg:flex flex-col fixed left-0 top-0 h-full z-40 transition-all duration-300',
        theme === 'dark' ? 'bg-gray-800 border-r border-gray-700' : 'bg-white border-r border-gray-200',
        sidebarCollapsed ? 'w-16' : 'w-64'
      )}>
        {/* Logo */}
        <div className={clsx(
          'h-16 flex items-center border-b',
          theme === 'dark' ? 'border-gray-700' : 'border-gray-200',
          sidebarCollapsed ? 'justify-center px-2' : 'justify-between px-4'
        )}>
          {!sidebarCollapsed && (
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center">
                <span className="text-white font-bold">H</span>
              </div>
              <span className="font-semibold text-lg">HealthTech</span>
            </div>
          )}
          <button
            onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
            className={clsx(
              'p-1.5 rounded-lg transition-colors',
              theme === 'dark' ? 'hover:bg-gray-700' : 'hover:bg-gray-100'
            )}
          >
            {sidebarCollapsed ? <IconChevronRight size={18} /> : <IconChevronLeft size={18} />}
          </button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 py-4 px-2 space-y-1">
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) => clsx(
                'flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors',
                isActive
                  ? 'bg-blue-600 text-white'
                  : theme === 'dark'
                    ? 'text-gray-300 hover:bg-gray-700'
                    : 'text-gray-600 hover:bg-gray-100',
                sidebarCollapsed && 'justify-center'
              )}
              title={sidebarCollapsed ? item.name : undefined}
            >
              <item.icon size={20} />
              {!sidebarCollapsed && <span>{item.name}</span>}
            </NavLink>
          ))}
        </nav>

        {/* User Section */}
        <div className={clsx(
          'p-4 border-t',
          theme === 'dark' ? 'border-gray-700' : 'border-gray-200'
        )}>
          {!sidebarCollapsed && userId && (
            <div className="flex items-center gap-2 mb-3 px-2">
              <div className={clsx(
                'w-8 h-8 rounded-full flex items-center justify-center',
                theme === 'dark' ? 'bg-gray-700' : 'bg-gray-200'
              )}>
                <IconUser size={16} />
              </div>
              <span className="text-sm truncate">{userId}</span>
            </div>
          )}
          <div className={clsx('flex gap-2', sidebarCollapsed ? 'flex-col' : '')}>
            <button
              onClick={toggleTheme}
              className={clsx(
                'flex-1 p-2 rounded-lg transition-colors flex items-center justify-center gap-2',
                theme === 'dark' ? 'hover:bg-gray-700' : 'hover:bg-gray-100'
              )}
              title={theme === 'dark' ? 'Light mode' : 'Dark mode'}
            >
              {theme === 'dark' ? <IconSun size={18} /> : <IconMoon size={18} />}
            </button>
            <button
              onClick={() => setLanguage(language === 'ar' ? 'en' : 'ar')}
              className={clsx(
                'flex-1 p-2 rounded-lg transition-colors text-sm font-medium',
                theme === 'dark' ? 'hover:bg-gray-700' : 'hover:bg-gray-100'
              )}
              title="Toggle language"
            >
              {language === 'ar' ? 'EN' : 'عربي'}
            </button>
          </div>
          {token && (
            <button
              onClick={handleLogout}
              className={clsx(
                'w-full mt-2 p-2 rounded-lg transition-colors flex items-center justify-center gap-2 text-red-500',
                theme === 'dark' ? 'hover:bg-gray-700' : 'hover:bg-gray-100'
              )}
            >
              <IconLogout size={18} />
              {!sidebarCollapsed && <span>{language === 'ar' ? 'تسجيل خروج' : 'Logout'}</span>}
            </button>
          )}
        </div>
      </aside>

      {/* Mobile Header */}
      <div className={clsx(
        'lg:hidden fixed top-0 left-0 right-0 h-14 z-50 flex items-center justify-between px-4',
        theme === 'dark' ? 'bg-gray-800 border-b border-gray-700' : 'bg-white border-b border-gray-200'
      )}>
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center">
            <span className="text-white font-bold text-sm">H</span>
          </div>
          <span className="font-semibold">HealthTech</span>
        </div>
        <button
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          className="p-2"
        >
          {mobileMenuOpen ? <IconX size={24} /> : <IconMenu2 size={24} />}
        </button>
      </div>

      {/* Mobile Menu Overlay */}
      {mobileMenuOpen && (
        <div
          className="lg:hidden fixed inset-0 bg-black/50 z-40"
          onClick={() => setMobileMenuOpen(false)}
        />
      )}

      {/* Mobile Sidebar */}
      <aside className={clsx(
        'lg:hidden fixed left-0 top-14 bottom-0 w-64 z-50 transform transition-transform duration-300',
        theme === 'dark' ? 'bg-gray-800' : 'bg-white',
        mobileMenuOpen ? 'translate-x-0' : '-translate-x-full'
      )}>
        <nav className="py-4 px-2 space-y-1">
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              onClick={() => setMobileMenuOpen(false)}
              className={({ isActive }) => clsx(
                'flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors',
                isActive
                  ? 'bg-blue-600 text-white'
                  : theme === 'dark'
                    ? 'text-gray-300 hover:bg-gray-700'
                    : 'text-gray-600 hover:bg-gray-100'
              )}
            >
              <item.icon size={20} />
              <span>{item.name}</span>
            </NavLink>
          ))}
        </nav>
        <div className={clsx(
          'absolute bottom-0 left-0 right-0 p-4 border-t',
          theme === 'dark' ? 'border-gray-700' : 'border-gray-200'
        )}>
          <div className="flex gap-2 mb-2">
            <button
              onClick={toggleTheme}
              className={clsx(
                'flex-1 p-2 rounded-lg transition-colors',
                theme === 'dark' ? 'hover:bg-gray-700' : 'hover:bg-gray-100'
              )}
            >
              {theme === 'dark' ? <IconSun size={18} /> : <IconMoon size={18} />}
            </button>
            <button
              onClick={() => setLanguage(language === 'ar' ? 'en' : 'ar')}
              className={clsx(
                'flex-1 p-2 rounded-lg transition-colors text-sm font-medium',
                theme === 'dark' ? 'hover:bg-gray-700' : 'hover:bg-gray-100'
              )}
            >
              {language === 'ar' ? 'EN' : 'عربي'}
            </button>
          </div>
          {token && (
            <button
              onClick={handleLogout}
              className="w-full p-2 rounded-lg text-red-500 hover:bg-red-500/10 flex items-center justify-center gap-2"
            >
              <IconLogout size={18} />
              <span>{language === 'ar' ? 'تسجيل خروج' : 'Logout'}</span>
            </button>
          )}
        </div>
      </aside>

      {/* Main Content */}
      <main className={clsx(
        'flex-1 min-h-screen transition-all duration-300',
        'lg:ml-64 pt-14 lg:pt-0',
        sidebarCollapsed && 'lg:ml-16'
      )}>
        <div className="p-4 lg:p-6">
          <Outlet />
        </div>
      </main>
    </div>
  );
};

export default ClinicalLayout;
