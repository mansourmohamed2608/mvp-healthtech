// gateway/src/llm/llm.service.spec.ts
import { Test, TestingModule } from '@nestjs/testing';
import { LlmService, LlmResponse, OrchestratePayload } from './llm.service';
import { InternalHttpClient } from '../http/internal-http-client.service';

describe('LlmService', () => {
  let service: LlmService;
  let mockAxiosClient: { post: jest.Mock };

  beforeEach(async () => {
    mockAxiosClient = {
      post: jest.fn(),
    };

    const mockHttpClient = {
      getClient: jest.fn(() => mockAxiosClient),
    };

    process.env.INTERNAL_SECRET = 'test-secret';
    process.env.LLM_SERVICE_URL = 'http://llm:5001';
    process.env.ORCHESTRATOR_URL = 'http://orchestrator:5006';

    const module: TestingModule = await Test.createTestingModule({
      providers: [
        LlmService,
        {
          provide: InternalHttpClient,
          useValue: mockHttpClient,
        },
      ],
    }).compile();

    service = module.get<LlmService>(LlmService);
  });

  afterEach(() => {
    delete process.env.INTERNAL_SECRET;
    delete process.env.LLM_SERVICE_URL;
    delete process.env.ORCHESTRATOR_URL;
  });

  describe('infer', () => {
    it('should call LLM service with message and sessionId', async () => {
      const expectedResponse: LlmResponse = {
        reply: 'أفهم مشكلتك، هل تريد حجز موعد؟',
        intent: 'booking_inquiry',
      };
      mockAxiosClient.post.mockResolvedValue({ data: expectedResponse });

      const result = await service.infer('عندي ألم', 'session-123');

      expect(result.reply).toBe(expectedResponse.reply);
      expect(mockAxiosClient.post).toHaveBeenCalledWith(
        '/infer',
        { message: 'عندي ألم', sessionId: 'session-123' },
        expect.any(Object),
      );
    });

    it('should handle timeout errors', async () => {
      mockAxiosClient.post.mockRejectedValue(new Error('ECONNABORTED'));

      await expect(service.infer('test', 'session')).rejects.toThrow();
    });
  });

  describe('chat', () => {
    it('should include conversation history', async () => {
      const response = { reply: 'OK', intent: 'confirm' };
      mockAxiosClient.post.mockResolvedValue({ data: response });

      await service.chat({
        message: 'نعم',
        history: [
          { role: 'user', content: 'عايز احجز' },
          { role: 'assistant', content: 'هل تريد قسم القلب؟' },
        ],
        sessionId: 'chat-1',
      });

      expect(mockAxiosClient.post).toHaveBeenCalledWith(
        '/chat',
        expect.objectContaining({
          history: expect.arrayContaining([
            expect.objectContaining({ role: 'user' }),
          ]),
        }),
        expect.any(Object),
      );
    });

    it('should pass intent for RAG retrieval', async () => {
      mockAxiosClient.post.mockResolvedValue({ data: { reply: 'test' } });

      await service.chat({
        message: 'ما هي ساعات العمل؟',
        sessionId: 'chat-2',
        intent: 'hours_inquiry',
      });

      expect(mockAxiosClient.post).toHaveBeenCalledWith(
        '/chat',
        expect.objectContaining({ intent: 'hours_inquiry' }),
        expect.any(Object),
      );
    });
  });

  describe('orchestrate', () => {
    it('should orchestrate VA flow with slots', async () => {
      const orchestrateResponse = {
        reply: 'تم الحجز',
        slots: { date: '2025-02-10', time: '14:00' },
      };
      mockAxiosClient.post.mockResolvedValue({ data: orchestrateResponse });

      const payload: OrchestratePayload = {
        transcript: 'احجز لي موعد بكره الساعة 2',
        sessionId: 'va-1',
        tenantId: 'clinic-1',
      };

      const result = await service.orchestrate(payload);

      expect(result.slots).toBeDefined();
      expect(result.slots?.date).toBe('2025-02-10');
    });

    it('should include dialect preference', async () => {
      mockAxiosClient.post.mockResolvedValue({ data: { reply: 'test' } });

      await service.orchestrate({
        transcript: 'test',
        sessionId: 'va-2',
        dialect: 'saudi',
      });

      expect(mockAxiosClient.post).toHaveBeenCalledWith(
        '/orchestrate',
        expect.objectContaining({ dialect: 'saudi' }),
        expect.any(Object),
      );
    });
  });

  describe('generateSoap', () => {
    it('should generate SOAP note from transcript', async () => {
      const soapResponse = {
        subjective: 'Patient reports chest pain',
        objective: 'Vitals stable',
        assessment: 'Suspected angina',
        plan: 'Order ECG, cardiac enzymes',
      };
      mockAxiosClient.post.mockResolvedValue({ data: soapResponse });

      const result = await service.generateSoap('Doctor: How can I help? Patient: I have chest pain...');

      expect(result.subjective).toBeDefined();
      expect(result.plan).toBeDefined();
    });
  });
});
