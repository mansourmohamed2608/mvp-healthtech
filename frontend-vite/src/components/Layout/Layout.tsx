import { Outlet } from 'react-router-dom';
import { motion } from 'framer-motion';
import Navbar from './Navbar';
import Footer from './Footer';
import CustomCursor from '@components/UI/CustomCursor';
import ScrollProgress from '@components/UI/ScrollProgress';
import BackgroundEffects from '@components/UI/BackgroundEffects';

const Layout = () => {
  return (
    <div className="relative min-h-screen grain-texture overflow-x-hidden">
      {/* Background Effects */}
      <BackgroundEffects />

      {/* Custom Cursor (Desktop only) */}
      <CustomCursor />

      {/* Scroll Progress */}
      <ScrollProgress />

      {/* Navigation */}
      <Navbar />

      {/* Main Content */}
      <motion.main
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.5 }}
        className="relative z-10"
      >
        <Outlet />
      </motion.main>

      {/* Footer */}
      <Footer />
    </div>
  );
};

export default Layout;
