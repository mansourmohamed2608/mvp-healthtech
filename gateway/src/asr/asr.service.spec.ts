// gateway/src/asr/asr.service.spec.ts
import { Test, TestingModule } from '@nestjs/testing';
import { AsrService, TranscriptionResponse, SpeakerRole } from './asr.service';
import { InternalHttpClient } from '../http/internal-http-client.service';

describe('AsrService', () => {
  let service: AsrService;
  let httpClient: jest.Mocked<InternalHttpClient>;
  let mockAxiosClient: { post: jest.Mock };

  const mockTranscriptionResponse: TranscriptionResponse = {
    text: 'مرحبا دكتور أنا عندي ألم في الصدر',
    confidence: 0.92,
    language: 'ar',
    segments: [
      { speaker: 'SPEAKER_00', text: 'مرحبا دكتور', start: 0.0, end: 1.5 },
      { speaker: 'SPEAKER_01', text: 'أهلا وسهلا', start: 1.6, end: 2.8 },
      { speaker: 'SPEAKER_00', text: 'أنا عندي ألم في الصدر', start: 3.0, end: 5.2 },
    ],
    speakers: ['SPEAKER_00', 'SPEAKER_01'],
  };

  const mockRoleResponse: { roles: SpeakerRole[] } = {
    roles: [
      { speaker_id: 'SPEAKER_00', role: 'Patient', confidence: 0.95, reasoning: 'Symptoms description' },
      { speaker_id: 'SPEAKER_01', role: 'Doctor', confidence: 0.88, reasoning: 'Greeting response' },
    ],
  };

  beforeEach(async () => {
    mockAxiosClient = {
      post: jest.fn(),
    };

    const mockHttpClient = {
      getClient: jest.fn(() => mockAxiosClient),
    };

    process.env.INTERNAL_SECRET = 'test-secret';
    process.env.ASR_SERVICE_URL = 'http://asr:5000';
    process.env.LLM_SERVICE_URL = 'http://llm:5001';

    const module: TestingModule = await Test.createTestingModule({
      providers: [
        AsrService,
        {
          provide: InternalHttpClient,
          useValue: mockHttpClient,
        },
      ],
    }).compile();

    service = module.get<AsrService>(AsrService);
    httpClient = module.get(InternalHttpClient);
  });

  afterEach(() => {
    delete process.env.INTERNAL_SECRET;
    delete process.env.ASR_SERVICE_URL;
    delete process.env.LLM_SERVICE_URL;
  });

  describe('transcribe', () => {
    it('should transcribe audio and identify speaker roles', async () => {
      mockAxiosClient.post
        .mockResolvedValueOnce({ data: mockTranscriptionResponse })
        .mockResolvedValueOnce({ data: mockRoleResponse });

      const result = await service.transcribe(
        'base64audio',
        'call-123',
        { identifySpeakers: true },
      );

      expect(result.text).toBe(mockTranscriptionResponse.text);
      expect(result.segments).toHaveLength(3);
      expect(mockAxiosClient.post).toHaveBeenCalledTimes(2);
    });

    it('should transcribe without speaker identification', async () => {
      const responseWithoutSegments = { ...mockTranscriptionResponse, segments: [] };
      mockAxiosClient.post.mockResolvedValueOnce({ data: responseWithoutSegments });

      const result = await service.transcribe(
        'base64audio',
        'call-456',
        { identifySpeakers: false },
      );

      expect(result.text).toBeDefined();
      expect(mockAxiosClient.post).toHaveBeenCalledTimes(1);
    });

    it('should pass dialect and language options', async () => {
      mockAxiosClient.post.mockResolvedValueOnce({ data: { text: 'test', segments: [] } });

      await service.transcribe('audio', 'call-789', {
        dialect: 'saudi',
        language: 'ar',
        enableDiarization: true,
      });

      expect(mockAxiosClient.post).toHaveBeenCalledWith(
        '/transcribe',
        expect.objectContaining({
          dialect: 'saudi',
          language: 'ar',
          enable_diarization: true,
        }),
        expect.any(Object),
      );
    });

    it('should handle transcription service errors', async () => {
      mockAxiosClient.post.mockRejectedValue(new Error('Service unavailable'));

      await expect(service.transcribe('audio', 'call-err')).rejects.toThrow('Service unavailable');
    });

    it('should handle speaker role identification failure gracefully', async () => {
      mockAxiosClient.post
        .mockResolvedValueOnce({ data: mockTranscriptionResponse })
        .mockRejectedValueOnce(new Error('LLM unavailable'));

      // Should still return transcription even if role ID fails
      const result = await service.transcribe('audio', 'call-123', { identifySpeakers: true });
      expect(result.text).toBeDefined();
    });
  });

  describe('stream', () => {
    it('should stream audio transcription', async () => {
      const streamResponse = { partial: 'مرحبا', final: 'مرحبا دكتور' };
      mockAxiosClient.post.mockResolvedValue({ data: streamResponse });

      const result = await service.stream('base64audio', 'call-stream');

      expect(result.final).toBe('مرحبا دكتور');
    });
  });
});
