// frontend-vite/cypress/support/commands.ts
/**
 * Cypress Custom Commands
 */

Cypress.Commands.add('login', (email: string, password: string) => {
  cy.request({
    method: 'POST',
    url: `${Cypress.env('apiUrl')}/auth/login`,
    body: { username: email, password },
  }).then((response) => {
    window.localStorage.setItem('auth_token', response.body.access_token);
  });
});

Cypress.Commands.add('getByTestId', (testId: string) => {
  return cy.get(`[data-testid="${testId}"]`);
});
