import { useEffect } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { AnimatePresence } from 'framer-motion';
import { useThemeStore } from '@store/themeStore';
import { useSmoothScroll } from '@hooks/useSmoothScroll';

// Layout
import Layout from '@components/Layout/Layout';

// Pages
import Home from '@pages/Home';
import Features from '@pages/Features';
import ClinicalNotes from '@pages/ClinicalNotes';
import VoiceTranscription from '@pages/VoiceTranscription';
import FHIRIntegration from '@pages/FHIRIntegration';
import SOAPGeneration from '@pages/SOAPGeneration';
import Dashboard from '@pages/Dashboard';
import About from '@pages/About';
import Pricing from '@pages/Pricing';
import Demo from '@pages/Demo';
import ServiceTest from '@pages/ServiceTest';
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
            <Route index element={<Home />} />
            <Route path="features" element={<Features />} />
            <Route path="features/clinical-notes" element={<ClinicalNotes />} />
            <Route path="features/voice-transcription" element={<VoiceTranscription />} />
            <Route path="features/fhir-integration" element={<FHIRIntegration />} />
            <Route path="features/soap-generation" element={<SOAPGeneration />} />
            <Route path="dashboard" element={<Dashboard />} />
            <Route path="voice-agent" element={<VoiceAgent />} />
            <Route path="about" element={<About />} />
            <Route path="pricing" element={<Pricing />} />
            <Route path="demo" element={<Demo />} />
            <Route path="test" element={<ServiceTest />} />
          </Route>
        </Routes>
      </AnimatePresence>
    </BrowserRouter>
  );
}

export default App;
