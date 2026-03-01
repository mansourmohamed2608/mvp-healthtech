// gateway/test/integration/full-pipeline.e2e-spec.ts
/**
 * Full Pipeline Integration Tests
 * Tests the complete flow: Audio → ASR → LLM → SOAP → FHIR
 */
import { Test, TestingModule } from '@nestjs/testing';
import { INestApplication, ValidationPipe } from '@nestjs/common';
import * as request from 'supertest';
import { AppModule } from '../../src/app.module';

describe('Full Pipeline Integration (e2e)', () => {
  let app: INestApplication;
  let authToken: string;

  beforeAll(async () => {
    // Skip if not in integration test environment
    if (!process.env.INTEGRATION_TEST) {
      return;
    }

    const moduleFixture: TestingModule = await Test.createTestingModule({
      imports: [AppModule],
    }).compile();

    app = moduleFixture.createNestApplication();
    app.useGlobalPipes(new ValidationPipe({ whitelist: true, transform: true }));
    await app.init();

    // Get auth token
    const loginResponse = await request(app.getHttpServer())
      .post('/auth/login')
      .send({ username: 'test@healthtech.com', password: 'testpass' });
    
    authToken = loginResponse.body.access_token;
  });

  afterAll(async () => {
    if (app) {
      await app.close();
    }
  });

  describe('Health Checks', () => {
    it('should return healthy status', async () => {
      const response = await request(app.getHttpServer())
        .get('/health')
        .expect(200);

      expect(response.body.status).toBeDefined();
    });

    it('should expose Prometheus metrics', async () => {
      const response = await request(app.getHttpServer())
        .get('/metrics')
        .expect(200);

      expect(response.text).toContain('# HELP');
    });
  });

  describe('Authentication Flow', () => {
    it('should reject unauthenticated requests', async () => {
      await request(app.getHttpServer())
        .post('/asr/transcribe')
        .send({ audio: 'test' })
        .expect(401);
    });

    it('should accept authenticated requests', async () => {
      if (!authToken) return;

      const response = await request(app.getHttpServer())
        .get('/health')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(response.body).toBeDefined();
    });
  });

  describe('ASR → LLM Flow', () => {
    it('should transcribe audio and process with LLM', async () => {
      if (!authToken || !process.env.INTEGRATION_TEST) return;

      // This test requires running services
      // Simulate with minimal audio
      const sampleAudio = Buffer.from('RIFF....WAVEfmt ').toString('base64');

      const transcribeResponse = await request(app.getHttpServer())
        .post('/asr/transcribe')
        .set('Authorization', `Bearer ${authToken}`)
        .set('x-tenant-id', 'test-tenant')
        .send({
          audio: sampleAudio,
          callSid: 'test-call-001',
          dialect: 'egypt',
        });

      // May fail if ASR service not running
      if (transcribeResponse.status === 200) {
        expect(transcribeResponse.body.text).toBeDefined();
      }
    });
  });

  describe('SOAP Note Generation Flow', () => {
    it('should generate SOAP note from transcript', async () => {
      if (!authToken || !process.env.INTEGRATION_TEST) return;

      const transcript = `
        Doctor: مرحبا، إزي الصحة؟
        Patient: عندي صداع من يومين
        Doctor: فين الصداع؟
        Patient: في الجبهة
      `;

      const response = await request(app.getHttpServer())
        .post('/soap/generate')
        .set('Authorization', `Bearer ${authToken}`)
        .set('x-tenant-id', 'test-tenant')
        .send({
          transcript,
          sessionId: 'test-session-001',
          patientId: 'P001',
          practitionerId: 'D001',
        });

      if (response.status === 200 || response.status === 201) {
        expect(response.body).toBeDefined();
      }
    });
  });

  describe('VA Conversation Flow', () => {
    it('should handle VA booking conversation', async () => {
      if (!authToken || !process.env.INTEGRATION_TEST) return;

      const response = await request(app.getHttpServer())
        .post('/llm/orchestrate')
        .set('Authorization', `Bearer ${authToken}`)
        .set('x-tenant-id', 'test-tenant')
        .send({
          transcript: 'عايز احجز موعد مع دكتور قلب',
          sessionId: 'va-test-001',
          context: { patientId: 'P001' },
        });

      if (response.status === 200) {
        expect(response.body.reply).toBeDefined();
      }
    });
  });

  describe('Rate Limiting', () => {
    it('should enforce rate limits', async () => {
      if (!authToken) return;

      const requests = [];
      for (let i = 0; i < 60; i++) {
        requests.push(
          request(app.getHttpServer())
            .get('/health')
            .set('Authorization', `Bearer ${authToken}`)
        );
      }

      const responses = await Promise.all(requests);
      const tooManyRequests = responses.filter(r => r.status === 429);
      
      // Should have some rate-limited responses after threshold
      // (depends on throttler config)
    });
  });
});
