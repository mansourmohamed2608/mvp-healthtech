import { useEffect } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AnimatePresence } from 'framer-motion';
import { useThemeStore } from '@store/themeStore';
import { useSmoothScroll } from '@hooks/useSmoothScroll';

// Layout
import Layout from '@components/Layout/Layout';

// Pages
import ClinicalNotes from '@pages/ClinicalNotes';
import VoiceAgent from '@pages/VoiceAgent';

function App() {
  const { theme } = useThemeStore();
  useSmoothScroll();

  useEffect(() => {
    // Apply theme class to document
    document.documentElement.classList.toggle('dark', theme === 'dark');
  }, [theme]);

  return (
    <BrowserRouter>
      <AnimatePresence mode="wait">
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<Navigate to="/voice-agent" replace />} />
            <Route path="voice-agent" element={<VoiceAgent />} />
            <Route path="clinical-notes" element={<ClinicalNotes />} />
            <Route path="features/clinical-notes" element={<Navigate to="/clinical-notes" replace />} />
            <Route path="*" element={<Navigate to="/voice-agent" replace />} />
          </Route>
        </Routes>
      </AnimatePresence>
    </BrowserRouter>
  );
}

export default App;
