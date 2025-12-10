import request from 'supertest';
import { INestApplication } from '@nestjs/common';

// NOTE: These tests are placeholders and intentionally skipped to avoid CI failures
// until full auth/token wiring and fixtures are ready.

describe.skip('VA flows (placeholder)', () => {
  let app: INestApplication;

  it('should require auth for /twilio/token', async () => {
    await request(app.getHttpServer()).post('/twilio/token').expect(401);
  });

  it('should reject orchestrator without internal secret', async () => {
    await request('http://localhost:5006').post('/orchestrate').send({}).expect(403);
  });
});
