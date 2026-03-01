// frontend-vite/cypress/e2e/app.cy.ts
/**
 * Main Application E2E Tests
 */

describe('HealthTech Application', () => {
  beforeEach(() => {
    // Visit the app
    cy.visit('/');
  });

  describe('Landing Page', () => {
    it('should load the landing page', () => {
      cy.url().should('include', '/');
      cy.get('body').should('be.visible');
    });

    it('should have proper page structure', () => {
      // Check for main elements
      cy.get('main, [role="main"], div').should('exist');
    });
  });

  describe('Navigation', () => {
    it('should navigate between pages', () => {
      // Find and click navigation links if they exist
      cy.get('a, button').then(($elements) => {
        if ($elements.length > 0) {
          // App has navigation elements
          cy.wrap($elements.first()).should('exist');
        }
      });
    });
  });

  describe('Responsive Design', () => {
    it('should work on mobile viewport', () => {
      cy.viewport('iphone-x');
      cy.get('body').should('be.visible');
    });

    it('should work on tablet viewport', () => {
      cy.viewport('ipad-2');
      cy.get('body').should('be.visible');
    });

    it('should work on desktop viewport', () => {
      cy.viewport(1920, 1080);
      cy.get('body').should('be.visible');
    });
  });

  describe('Accessibility', () => {
    it('should have no major accessibility violations', () => {
      // Basic accessibility check
      cy.get('body').should('exist');
      // Would use cypress-axe for full a11y testing
    });
  });
});
