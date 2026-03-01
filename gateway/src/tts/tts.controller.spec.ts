// gateway/src/tts/tts.controller.spec.ts
import { Test, TestingModule } from '@nestjs/testing';
import { TtsController } from './tts.controller';
import { TtsService } from './tts.service';
import { JwtAuthGuard } from '../auth/jwt.guard';
import { TenantGuard } from '../auth/tenant.guard';

describe('TtsController', () => {
  let controller: TtsController;
  let ttsService: jest.Mocked<TtsService>;

  beforeEach(async () => {
    const mockTtsService = {
      synthesize: jest.fn(),
    };

    const module: TestingModule = await Test.createTestingModule({
      controllers: [TtsController],
      providers: [
        {
          provide: TtsService,
          useValue: mockTtsService,
        },
      ],
    })
      .overrideGuard(JwtAuthGuard)
      .useValue({ canActivate: () => true })
      .overrideGuard(TenantGuard)
      .useValue({ canActivate: () => true })
      .compile();

    controller = module.get<TtsController>(TtsController);
    ttsService = module.get(TtsService);
  });

  describe('synthesize', () => {
    it('should synthesize text to audio', async () => {
      const mockAudio = 'base64encodedaudio';
      ttsService.synthesize.mockResolvedValue({
        audioBase64: mockAudio,
        format: 'mulaw',
      });

      const dto = { text: 'مرحبا، كيف يمكنني مساعدتك؟' };
      const mockRequest = { headers: {} };

      const result = await controller.synthesize(dto, mockRequest);

      expect(result.audio).toBe(mockAudio);
      expect(result.format).toBe('mulaw');
      expect(ttsService.synthesize).toHaveBeenCalledWith(dto.text);
    });

    it('should handle empty text', async () => {
      ttsService.synthesize.mockResolvedValue({
        audioBase64: '',
        format: 'mulaw',
      });

      const dto = { text: '' };
      const mockRequest = { headers: {} };

      const result = await controller.synthesize(dto, mockRequest);

      expect(result.audio).toBe('');
    });

    it('should propagate service errors', async () => {
      ttsService.synthesize.mockRejectedValue(new Error('TTS service down'));

      const dto = { text: 'test' };
      const mockRequest = { headers: {} };

      await expect(controller.synthesize(dto, mockRequest)).rejects.toThrow();
    });

    it('should handle Arabic text correctly', async () => {
      const arabicText = 'أهلا وسهلا، موعدك تم تأكيده ليوم الاثنين';
      ttsService.synthesize.mockResolvedValue({
        audioBase64: 'audio-data',
        format: 'mulaw',
      });

      const dto = { text: arabicText };
      const mockRequest = { headers: {} };

      await controller.synthesize(dto, mockRequest);

      expect(ttsService.synthesize).toHaveBeenCalledWith(arabicText);
    });

    it('should handle voice parameter', async () => {
      ttsService.synthesize.mockResolvedValue({
        audioBase64: 'audio',
        format: 'wav',
      });

      const dto = { text: 'test', voice: 'egtts' };
      const mockRequest = { headers: {} };

      await controller.synthesize(dto, mockRequest);

      expect(ttsService.synthesize).toHaveBeenCalledWith(dto.text);
    });
  });
});
