// gateway/src/llm/llm.controller.spec.ts
import { Test, TestingModule } from '@nestjs/testing';
import { LlmController } from './llm.controller';
import { LlmService, LlmResponse, OrchestrateResponse } from './llm.service';
import { JwtAuthGuard } from '../auth/jwt.guard';
import { TenantGuard } from '../auth/tenant.guard';

describe('LlmController', () => {
  let controller: LlmController;
  let llmService: jest.Mocked<LlmService>;

  const mockLlmResponse: LlmResponse = {
    reply: 'أفهم أنك تعاني من ألم في الصدر. متى بدأ هذا الألم؟',
    intent: 'symptoms_inquiry',
    confidence: 0.88,
  };

  const mockOrchestrateResponse: OrchestrateResponse = {
    reply: 'حجزنا لك موعد يوم الثلاثاء الساعة 10',
    intent: 'booking_confirmed',
    confidence: 0.92,
    slots: {
      date: '2025-02-11',
      time: '10:00',
      department: 'cardiology',
    },
  };

  beforeEach(async () => {
    const mockLlmService = {
      infer: jest.fn(),
      chat: jest.fn(),
      orchestrate: jest.fn(),
      generateSoap: jest.fn(),
    };

    const module: TestingModule = await Test.createTestingModule({
      controllers: [LlmController],
      providers: [
        {
          provide: LlmService,
          useValue: mockLlmService,
        },
      ],
    })
      .overrideGuard(JwtAuthGuard)
      .useValue({ canActivate: () => true })
      .overrideGuard(TenantGuard)
      .useValue({ canActivate: () => true })
      .compile();

    controller = module.get<LlmController>(LlmController);
    llmService = module.get(LlmService);
  });

  describe('infer', () => {
    it('should return LLM inference response', async () => {
      llmService.infer.mockResolvedValue(mockLlmResponse);

      const dto = {
        message: 'عندي ألم في صدري',
        sessionId: 'session-123',
      };
      const mockRequest = { headers: {} } as any;

      const result = await controller.infer(dto, mockRequest);

      expect(result.reply).toBeDefined();
      expect(result.intent).toBe('symptoms_inquiry');
      expect(llmService.infer).toHaveBeenCalledWith(dto.message, dto.sessionId);
    });

    it('should handle empty messages', async () => {
      llmService.infer.mockResolvedValue({ reply: '', intent: 'unknown' });

      const dto = { message: '', sessionId: 'session-456' };
      const mockRequest = { headers: {} } as any;

      const result = await controller.infer(dto, mockRequest);
      expect(result.reply).toBe('');
    });

    it('should propagate service errors', async () => {
      llmService.infer.mockRejectedValue(new Error('Model timeout'));

      const dto = { message: 'test', sessionId: 'session-err' };
      const mockRequest = { headers: {} } as any;

      await expect(controller.infer(dto, mockRequest)).rejects.toThrow();
    });
  });

  describe('orchestrate', () => {
    it('should orchestrate VA conversation flow', async () => {
      llmService.orchestrate.mockResolvedValue(mockOrchestrateResponse);

      const dto = {
        transcript: 'عايز احجز موعد مع دكتور قلب',
        sessionId: 'va-session-1',
        context: { patientId: 'P123' },
      };
      const mockRequest = { 
        headers: { 'x-tenant-id': 'tenant-1' },
        user: { tenantId: 'tenant-1' },
      } as any;

      const result = await controller.orchestrate(dto, mockRequest);

      expect(result.slots).toBeDefined();
      expect(result.slots?.department).toBe('cardiology');
    });

    it('should pass tenant context to orchestrator', async () => {
      llmService.orchestrate.mockResolvedValue(mockOrchestrateResponse);

      const dto = {
        transcript: 'test',
        sessionId: 'va-session-2',
        tenantId: 'clinic-abc',
      };
      const mockRequest = { 
        headers: { 'x-tenant-id': 'clinic-abc' },
        user: { tenantId: 'clinic-abc' },
      } as any;

      await controller.orchestrate(dto, mockRequest);

      expect(llmService.orchestrate).toHaveBeenCalledWith(
        expect.objectContaining({
          tenantId: 'clinic-abc',
        }),
      );
    });
  });

  describe('chat', () => {
    it('should handle chat with history', async () => {
      llmService.chat.mockResolvedValue(mockLlmResponse);

      const dto = {
        message: 'شكرا دكتور',
        history: [
          { role: 'user', content: 'عندي صداع' },
          { role: 'assistant', content: 'متى بدأ؟' },
        ],
        sessionId: 'chat-session-1',
      };
      const mockRequest = { headers: {} } as any;

      const result = await controller.chat(dto, mockRequest);

      expect(result.reply).toBeDefined();
      expect(llmService.chat).toHaveBeenCalledWith(
        expect.objectContaining({
          history: expect.arrayContaining([
            expect.objectContaining({ role: 'user' }),
          ]),
        }),
      );
    });
  });

  describe('generateSoap', () => {
    it('should generate SOAP note from transcript', async () => {
      const soapResponse = {
        reply: 'SOAP Note generated',
        subjective: 'Patient reports chest pain',
        objective: 'BP 120/80',
        assessment: 'Possible angina',
        plan: 'Order ECG',
      };
      llmService.generateSoap.mockResolvedValue(soapResponse);

      const dto = { transcript: 'Patient: عندي ألم في صدري...' };
      const mockRequest = { headers: {} } as any;

      const result = await controller.generateSoap(dto, mockRequest);

      expect(result).toBeDefined();
      expect(llmService.generateSoap).toHaveBeenCalledWith(dto.transcript);
    });
  });
});
