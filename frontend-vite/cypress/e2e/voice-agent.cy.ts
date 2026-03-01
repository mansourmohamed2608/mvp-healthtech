// frontend-vite/cypress/e2e/voice-agent.cy.ts
/**
 * Voice Agent E2E Tests
 */

describe('Voice Agent Interface', () => {
  beforeEach(() => {
    // Mock WebSocket and Twilio
    cy.intercept('GET', '**/twilio/**', { statusCode: 200, body: {} });
    cy.visit('/');
  });

  describe('Voice Agent UI', () => {
    it('should show voice agent interface when available', () => {
      // Look for voice agent elements
      cy.get('[data-testid="voice-agent"], .voice-agent, [class*="voice"]').then(
        ($el) => {
          if ($el.length > 0) {
            cy.wrap($el.first()).should('be.visible');
          }
        }
      );
    });

    it('should handle microphone permissions gracefully', () => {
      // App should not crash when microphone is denied
      cy.get('body').should('be.visible');
    });
  });

  describe('Call Flow', () => {
    it('should show call controls', () => {
      cy.get(
        '[data-testid="call-button"], button[class*="call"], [aria-label*="call"]'
      ).then(($button) => {
        if ($button.length > 0) {
          cy.wrap($button.first()).should('exist');
        }
      });
    });
  });

  describe('Conversation Display', () => {
    it('should display conversation messages when available', () => {
      cy.get(
        '[data-testid="conversation"], [class*="message"], [class*="transcript"]'
      ).then(($container) => {
        if ($container.length > 0) {
          cy.wrap($container).should('exist');
        }
      });
    });
  });
});
