import { Test, TestingModule } from '@nestjs/testing';
import { INestApplication } from '@nestjs/common';
import request from 'supertest';
import axios from 'axios';
import { JwtService } from '@nestjs/jwt';
import { AsrController } from './asr/asr.controller';
import { LlmController } from './llm/llm.controller';
import { TtsController } from './tts/tts.controller';
import { SoapController } from './soap/soap.controller';
import { FhirController } from './fhir/fhir.controller';
import { AsrService } from './asr/asr.service';
import { LlmService } from './llm/llm.service';
import { TtsService } from './tts/tts.service';
import { WsJwtGuard } from './auth/ws-jwt.guard';
import { JwtAuthGuard } from './auth/jwt.guard';
import { APP_GUARD } from '@nestjs/core';
import { AuditService } from './audit/audit.service';

jest.mock('./auth/jwt.guard', () => ({
  JwtAuthGuard: jest.fn().mockImplementation(() => ({ canActivate: () => true })),
}));
jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

const signJwt = () => {
  const secret = process.env.JWT_SECRET || 'test-secret';
  const jwtService = new JwtService({ secret });
  return jwtService.sign({ sub: 'tester', roles: ['clinician'] });
};

describe('Gateway contract (mocked downstream)', () => {
  let app: INestApplication;
  beforeAll(async () => {
    process.env.INTERNAL_SECRET = process.env.INTERNAL_SECRET || 'test-internal';
    const jwtService = new JwtService({ secret: process.env.JWT_SECRET || 'test-secret' });
    const moduleFixture: TestingModule = await Test.createTestingModule({
      controllers: [AsrController, LlmController, TtsController, SoapController, FhirController],
      providers: [
        { provide: AsrService, useValue: { transcribe: jest.fn().mockResolvedValue({ text: 'hello', segments: [], rtf: 0.1 }) } },
        { provide: LlmService, useValue: { chat: jest.fn().mockResolvedValue({ reply: 'hi', intent: 'general', totalLatencyMs: 10 }) } },
        { provide: TtsService, useValue: { synthesize: jest.fn().mockResolvedValue({ audioBase64: 'YmFzZTY0', format: 'mulaw', sampleRate: 8000 }) } },
        { provide: JwtService, useValue: jwtService },
        { provide: WsJwtGuard, useValue: { canActivate: () => true } },
        { provide: JwtAuthGuard, useValue: { canActivate: () => true } },
        { provide: APP_GUARD, useValue: { canActivate: () => true } },
        { provide: AuditService, useValue: { log: jest.fn() } },
      ],
    }).compile();
    app = moduleFixture.createNestApplication();
    await app.init();
  });

  afterAll(async () => {
    await app.close();
  });

  beforeEach(() => {
    mockedAxios.post.mockReset();
    mockedAxios.get.mockReset();
    mockedAxios.patch?.mockReset?.();
  });

  const auth = () => ({ Authorization: `Bearer ${signJwt()}` });

  it('/asr/transcribe returns camelCase', async () => {
    mockedAxios.post.mockResolvedValueOnce({ data: { text: 'hello', segments: [], rtf: 0.1 } });
    const res = await request(app.getHttpServer())
      .post('/asr/transcribe')
      .set(auth())
      .send({ audio: 'YmFzZTY0', callSid: 'call1' })
      .expect((r) => {
        if (![200, 201].includes(r.status)) {
          throw new Error(`Unexpected status ${r.status}`);
        }
      });
    expect(res.body.text).toBe('hello');
    expect(res.body.rtf).toBeDefined();
    expect(res.body.segments).toEqual([]);
  });

  it('/llm/chat returns reply', async () => {
    mockedAxios.post.mockResolvedValueOnce({ data: { reply: 'hi', intent: 'general', totalLatencyMs: 10 } });
    const res = await request(app.getHttpServer())
      .post('/llm/chat')
      .set(auth())
      .send({ message: 'hi', sessionId: 's1' })
      .expect((r) => {
        if (![200, 201].includes(r.status)) {
          throw new Error(`Unexpected status ${r.status}`);
        }
      });
    expect(res.body.reply).toBe('hi');
    expect(res.body.totalLatencyMs).toBeDefined();
    expect(res.body.intent).toBe('general');
  });

  it('/soap/generate and approve', async () => {
    mockedAxios.post
      .mockResolvedValueOnce({ data: { id: 'n1', status: 'pending', session_id: 's1', patient_id: 'p1', clinician_id: 'c1' } }) // SOAP generate
      .mockResolvedValueOnce({ data: { id: 'n1', status: 'approved', session_id: 's1' } }) // SOAP approve
      .mockResolvedValueOnce({ data: { documentReferenceId: 'doc1', encounterId: 'enc1', success: true } }); // FHIR write
    mockedAxios.patch?.mockResolvedValueOnce({ data: { id: 'n1', status: 'approved', session_id: 's1', patient_id: 'p1', clinician_id: 'c1', encounter_id: 'enc1' } });
    mockedAxios.get.mockResolvedValueOnce({ data: { notes: [] } });

    const gen = await request(app.getHttpServer())
      .post('/soap/generate')
      .set(auth())
      .send({ transcript: 't', sessionId: 's1', patientId: 'p1', practitionerId: 'c1' })
      .expect((r) => {
        if (![200, 201].includes(r.status)) {
          throw new Error(`Unexpected status ${r.status}`);
        }
      });
    expect(gen.body.id).toBe('n1');
    expect(gen.body.status).toBe('pending');

    const approve = await request(app.getHttpServer())
      .patch('/soap/notes/n1/approve')
      .set(auth())
      .expect((r) => {
        if (![200, 201].includes(r.status)) {
          throw new Error(`Unexpected status ${r.status}`);
        }
      });
    expect(approve.body.status).toBe('approved');
    // Verify FHIR was called with idempotency header
    const fhirCall = mockedAxios.post.mock.calls.find(([url]) => `${url}`.includes('/write'));
    expect(fhirCall?.[2]?.headers?.['Idempotency-Key']).toBeDefined();
  });

  it('/fhir/Patient proxied', async () => {
    mockedAxios.post.mockResolvedValueOnce({ data: { id: 'pat1' } });
    const res = await request(app.getHttpServer())
      .post('/fhir/Patient')
      .set(auth())
      .send({ resourceType: 'Patient' })
      .expect((r) => {
        if (![200, 201].includes(r.status)) {
          throw new Error(`Unexpected status ${r.status}`);
        }
      });
    expect(res.body.id).toBe('pat1');
  });
});
