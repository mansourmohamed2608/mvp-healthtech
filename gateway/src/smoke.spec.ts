import { Test, TestingModule } from '@nestjs/testing';
import { INestApplication, APP_GUARD, CanActivate, ExecutionContext } from '@nestjs/common';
import request from 'supertest';
import { AsrController } from './asr/asr.controller';
import { LlmController } from './llm/llm.controller';
import { TtsController } from './tts/tts.controller';
import { AsrService } from './asr/asr.service';
import { LlmService } from './llm/llm.service';
import { TtsService } from './tts/tts.service';
import { JwtAuthGuard } from './auth/jwt.guard';
import { RolesGuard } from './auth/roles.guard';

describe('Gateway smoke tests (mocked services)', () => {
  let app: INestApplication;

  class TestGuard implements CanActivate {
    canActivate(_context: ExecutionContext) {
      return true;
    }
  }

  beforeAll(async () => {
    const moduleBuilder = Test.createTestingModule({
      controllers: [AsrController, LlmController, TtsController],
      providers: [
        { provide: AsrService, useValue: { transcribe: jest.fn().mockResolvedValue({ text: 'hello world' }), stream: jest.fn() } },
        { provide: LlmService, useValue: { infer: jest.fn(), chat: jest.fn().mockResolvedValue({ reply: 'chat reply', intent: 'general' }) } },
        { provide: TtsService, useValue: { synthesize: jest.fn().mockResolvedValue({ audioBase64: 'YmFzZTY0', format: 'mulaw' }) } },
      ],
    })
      .overrideGuard(JwtAuthGuard)
      .useValue(new TestGuard() as any)
      .overrideGuard(RolesGuard)
      .useValue(new TestGuard() as any);

    const moduleFixture = await moduleBuilder.compile();

    app = moduleFixture.createNestApplication();
    await app.init();
  });

  afterAll(async () => {
    await app.close();
  });

  it('POST /asr/transcribe returns text', async () => {
    await request(app.getHttpServer())
      .post('/asr/transcribe')
      .send({ audio: 'XXX' })
      .expect((res) => expect([200, 201]).toContain(res.status))
      .expect((res) => expect(res.body.text).toBe('hello world'));
  });

  it('POST /llm/chat returns reply', async () => {
    await request(app.getHttpServer())
      .post('/llm/chat')
      .send({ message: 'hi', sessionId: 's1' })
      .expect((res) => expect([200, 201]).toContain(res.status))
      .expect((res) => expect(res.body.reply).toBe('chat reply'));
  });

  it('POST /tts/synthesize returns audio payload', async () => {
    await request(app.getHttpServer())
      .post('/tts/synthesize')
      .send({ text: 'hello' })
      .expect((res) => expect([200, 201]).toContain(res.status))
      .expect((res) => expect(res.body.audio).toBeDefined());
  });
});
