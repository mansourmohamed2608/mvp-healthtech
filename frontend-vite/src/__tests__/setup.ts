// frontend-vite/src/__tests__/setup.ts
/**
 * Test setup for React Testing Library and Vitest
 */
import '@testing-library/jest-dom';
import { vi } from 'vitest';

// Mock window.matchMedia
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation((query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
});

// Mock IntersectionObserver
class MockIntersectionObserver {
  observe = vi.fn();
  disconnect = vi.fn();
  unobserve = vi.fn();
}

Object.defineProperty(window, 'IntersectionObserver', {
  writable: true,
  value: MockIntersectionObserver,
});

// Mock ResizeObserver
class MockResizeObserver {
  observe = vi.fn();
  disconnect = vi.fn();
  unobserve = vi.fn();
}

Object.defineProperty(window, 'ResizeObserver', {
  writable: true,
  value: MockResizeObserver,
});

// Mock Twilio Voice SDK
vi.mock('@twilio/voice-sdk', () => ({
  Device: vi.fn().mockImplementation(() => ({
    register: vi.fn(),
    connect: vi.fn(),
    disconnectAll: vi.fn(),
    on: vi.fn(),
    off: vi.fn(),
  })),
}));

// Mock three.js (heavy dependency)
vi.mock('three', () => ({
  Scene: vi.fn(),
  PerspectiveCamera: vi.fn(),
  WebGLRenderer: vi.fn().mockImplementation(() => ({
    setSize: vi.fn(),
    render: vi.fn(),
    domElement: document.createElement('canvas'),
  })),
}));

// Mock react-three-fiber
vi.mock('@react-three/fiber', () => ({
  Canvas: ({ children }: any) => children,
  useFrame: vi.fn(),
}));

// Mock react-three-drei
vi.mock('@react-three/drei', () => ({
  OrbitControls: () => null,
  PerspectiveCamera: () => null,
}));
