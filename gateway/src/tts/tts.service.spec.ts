// gateway/src/tts/tts.service.spec.ts
import { Test, TestingModule } from '@nestjs/testing';
import { TtsService } from './tts.service';
import { InternalHttpClient } from '../http/internal-http-client.service';

describe('TtsService', () => {
  let service: TtsService;
  let mockAxiosClient: { post: jest.Mock };

  beforeEach(async () => {
    mockAxiosClient = {
      post: jest.fn(),
    };

    const mockHttpClient = {
      getClient: jest.fn(() => mockAxiosClient),
    };

    process.env.INTERNAL_SECRET = 'test-secret';
    process.env.TTS_SERVICE_URL = 'http://tts:5002';

    const module: TestingModule = await Test.createTestingModule({
      providers: [
        TtsService,
        {
          provide: InternalHttpClient,
          useValue: mockHttpClient,
        },
      ],
    }).compile();

    service = module.get<TtsService>(TtsService);
  });

  afterEach(() => {
    delete process.env.INTERNAL_SECRET;
    delete process.env.TTS_SERVICE_URL;
  });

  describe('synthesize', () => {
    it('should synthesize text to audio', async () => {
      const mockResponse = {
        audio: 'base64audio',
        duration: 2.5,
        sampleRate: 8000,
        format: 'mulaw',
      };
      mockAxiosClient.post.mockResolvedValue({ data: mockResponse });

      const result = await service.synthesize('مرحبا');

      expect(result.audioBase64).toBeDefined();
      expect(mockAxiosClient.post).toHaveBeenCalledWith(
        '/synthesize',
        expect.objectContaining({ text: 'مرحبا' }),
        expect.any(Object),
      );
    });

    it('should pass voice parameter', async () => {
      mockAxiosClient.post.mockResolvedValue({
        data: { audio: 'audio', duration: 1, sampleRate: 8000 },
      });

      await service.synthesize('test', 'session-1', 'saudi-tts');

      expect(mockAxiosClient.post).toHaveBeenCalledWith(
        '/synthesize',
        expect.objectContaining({
          voice: 'saudi-tts',
          sessionId: 'session-1',
        }),
        expect.any(Object),
      );
    });

    it('should handle long text', async () => {
      const longText = 'مرحبا '.repeat(100);
      mockAxiosClient.post.mockResolvedValue({
        data: { audio: 'audio', duration: 30, sampleRate: 8000 },
      });

      const result = await service.synthesize(longText);

      expect(result.audioBase64).toBeDefined();
    });

    it('should handle service errors', async () => {
      mockAxiosClient.post.mockRejectedValue(new Error('Service unavailable'));

      await expect(service.synthesize('test')).rejects.toThrow();
    });

    it('should include correlation ID in request', async () => {
      mockAxiosClient.post.mockResolvedValue({
        data: { audio: 'audio', duration: 1, sampleRate: 8000 },
      });

      await service.synthesize('test');

      expect(mockAxiosClient.post).toHaveBeenCalledWith(
        expect.any(String),
        expect.any(Object),
        expect.objectContaining({
          headers: expect.objectContaining({
            'x-correlation-id': expect.any(String),
          }),
        }),
      );
    });
  });
});
