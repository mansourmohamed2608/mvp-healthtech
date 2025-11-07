import { motion } from 'framer-motion';
import { useScrollProgress } from '@hooks/useScrollProgress';

const ScrollProgress = () => {
  const scrollProgress = useScrollProgress();

  return (
    <>
      {/* Top Progress Bar */}
      <motion.div
        className="fixed top-0 left-0 right-0 h-1 bg-gradient-to-r from-accent-500 via-accent-400 to-accent-600 origin-left z-[100]"
        style={{ scaleX: scrollProgress / 100 }}
        initial={{ scaleX: 0 }}
      />

      {/* Animated Gradient Glow */}
      <motion.div
        className="fixed top-0 left-0 h-1 w-32 bg-gradient-to-r from-transparent via-white to-transparent opacity-50 blur-sm z-[101]"
        style={{ left: `${scrollProgress}%` }}
        animate={{
          opacity: [0.3, 0.6, 0.3],
        }}
        transition={{
          duration: 2,
          repeat: Infinity,
          ease: 'easeInOut',
        }}
      />
    </>
  );
};

export default ScrollProgress;
