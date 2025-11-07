import { useRef, useEffect, useState } from 'react';
import { motion, useSpring, useTransform } from 'framer-motion';
import { useMousePosition } from '@hooks/useMousePosition';
import { useIsDesktop, usePrefersReducedMotion } from '@hooks/useMediaQuery';

const CustomCursor = () => {
  const cursorRef = useRef<HTMLDivElement>(null);
  const [isHovering, setIsHovering] = useState(false);
  const mousePosition = useMousePosition();
  const isDesktop = useIsDesktop();
  const prefersReducedMotion = usePrefersReducedMotion();

  const springConfig = { damping: 25, stiffness: 700 };
  const cursorX = useSpring(mousePosition.x, springConfig);
  const cursorY = useSpring(mousePosition.y, springConfig);

  // Create transforms before any conditional returns
  const mainCursorX = useTransform(cursorX, (x) => x - 8);
  const mainCursorY = useTransform(cursorY, (y) => y - 8);
  const trailCursorX = useTransform(cursorX, (x) => x - 20);
  const trailCursorY = useTransform(cursorY, (y) => y - 20);

  useEffect(() => {
    cursorX.set(mousePosition.x);
    cursorY.set(mousePosition.y);
  }, [mousePosition, cursorX, cursorY]);

  useEffect(() => {
    const handleMouseOver = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      const isInteractive = !!(
        target.tagName === 'BUTTON' ||
        target.tagName === 'A' ||
        target.closest('button') ||
        target.closest('a') ||
        target.classList.contains('magnetic-btn') ||
        target.classList.contains('card-3d')
      );

      setIsHovering(isInteractive);
    };

    document.addEventListener('mouseover', handleMouseOver);
    return () => document.removeEventListener('mouseover', handleMouseOver);
  }, []);

  // Don't render on mobile or if user prefers reduced motion
  if (!isDesktop || prefersReducedMotion) return null;

  return (
    <>
      {/* Main Cursor */}
      <motion.div
        ref={cursorRef}
        className="fixed top-0 left-0 pointer-events-none z-[9999] mix-blend-difference"
        style={{
          x: mainCursorX,
          y: mainCursorY,
        }}
      >
        <motion.div
          animate={{
            scale: isHovering ? 1.5 : 1,
            opacity: isHovering ? 0.8 : 1,
          }}
          transition={{ duration: 0.2 }}
          className="w-4 h-4 rounded-full bg-white"
        />
      </motion.div>

      {/* Cursor Trail */}
      <motion.div
        className="fixed top-0 left-0 pointer-events-none z-[9998]"
        style={{
          x: trailCursorX,
          y: trailCursorY,
        }}
      >
        <motion.div
          animate={{
            scale: isHovering ? 2 : 1,
          }}
          transition={{ duration: 0.4, ease: 'easeOut' }}
          className="w-10 h-10 rounded-full border-2 border-white/30"
        />
      </motion.div>
    </>
  );
};

export default CustomCursor;
