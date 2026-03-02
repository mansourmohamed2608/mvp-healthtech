import { useEffect } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { useThemeStore } from '@store/themeStore';
import { useAuthStore } from '@store/authStore';

// Layouts
import ClinicalLayout from '@components/Layout/ClinicalLayout';

// Pages
import Login from '@pages/Login';
import OidcCallback from '@pages/OidcCallback';
import AuthError from '@pages/AuthError';
import DashboardNew from '@pages/DashboardNew';
import ClinicalNotes from '@pages/ClinicalNotes';
import VoiceAgentClean from '@pages/VoiceAgentClean';
import KnowledgeBase from '@pages/KnowledgeBase';

// Protected Route wrapper
const ProtectedRoute = ({ children }: { children: React.ReactNode }) => {
  const { token } = useAuthStore();
  if (!token) {
    return <Navigate to="/login" replace />;
  }
  return <>{children}</>;
};

function App() {
  const { theme } = useThemeStore();

  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark');
  }, [theme]);

  return (
    <BrowserRouter>
      <Routes>
        {/* Public routes */}
        <Route path="/login" element={<Login />} />
        <Route path="/auth/callback" element={<OidcCallback />} />
        <Route path="/auth/error" element={<AuthError />} />
        
        {/* Protected routes with clinical layout */}
        <Route path="/" element={
          <ProtectedRoute>
            <ClinicalLayout />
          </ProtectedRoute>
        }>
          <Route index element={<Navigate to="/dashboard" replace />} />
          <Route path="dashboard" element={<DashboardNew />} />
          <Route path="voice-agent" element={<VoiceAgentClean />} />
          <Route path="clinical-notes" element={<ClinicalNotes />} />
          <Route path="knowledge-base" element={<KnowledgeBase />} />
          <Route path="*" element={<Navigate to="/dashboard" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
