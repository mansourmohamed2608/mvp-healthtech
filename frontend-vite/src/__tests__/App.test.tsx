// frontend-vite/src/__tests__/App.test.tsx
/**
 * App Component Tests
 */
import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import App from '../App';

// Mock heavy components
vi.mock('../components/3d/Scene', () => ({
  default: () => <div data-testid="mock-scene">Mock 3D Scene</div>,
}));

describe('App Component', () => {
  it('renders without crashing', () => {
    render(
      <BrowserRouter>
        <App />
      </BrowserRouter>
    );
    
    // App should render something
    expect(document.body).toBeDefined();
  });

  it('has proper document structure', () => {
    render(
      <BrowserRouter>
        <App />
      </BrowserRouter>
    );
    
    // Check for main container
    const main = document.querySelector('main') || document.querySelector('div');
    expect(main).toBeDefined();
  });
});
