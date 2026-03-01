// gateway/src/asr/asr.controller.spec.ts
import { Test, TestingModule } from '@nestjs/testing';
import { AsrController } from './asr.controller';
import { AsrService, TranscriptionResponse } from './asr.service';
import { JwtAuthGuard } from '../auth/jwt.guard';
import { TenantGuard } from '../auth/tenant.guard';
import { HttpException, HttpStatus } from '@nestjs/common';

describe('AsrController', () => {
  let controller: AsrController;
  let asrService: jest.Mocked<AsrService>;

  const mockTranscriptionResponse: TranscriptionResponse = {
    text: 'مرحبا دكتور',
    confidence: 0.95,
    language: 'ar',
    segments: [
      { speaker: 'SPEAKER_00', text: 'مرحبا دكتور', start: 0.0, end: 1.5 },
    ],
    speakers: ['SPEAKER_00'],
    roles: [
      {
        speaker_id: 'SPEAKER_00',
        role: 'Patient',
        confidence: 0.9,
        reasoning: 'Greeting pattern',
      },
    ],
  };

  beforeEach(async () => {
    const mockAsrService = {
      transcribe: jest.fn(),
      stream: jest.fn(),
    };

    const module: TestingModule = await Test.createTestingModule({
      controllers: [AsrController],
      providers: [
        {
          provide: AsrService,
          useValue: mockAsrService,
        },
      ],
    })
      .overrideGuard(JwtAuthGuard)
      .useValue({ canActivate: () => true })
      .overrideGuard(TenantGuard)
      .useValue({ canActivate: () => true })
      .compile();

    controller = module.get<AsrController>(AsrController);
    asrService = module.get(AsrService);
  });

  describe('transcribe', () => {
    it('should successfully transcribe audio', async () => {
      asrService.transcribe.mockResolvedValue(mockTranscriptionResponse);

      const dto = {
        audio: 'base64encodedaudio',
        callSid: 'CA123456',
        dialect: 'egypt',
      };

      const mockRequest = {
        headers: { 'x-correlation-id': 'test-corr-id' },
      } as any;

      const result = await controller.transcribe(dto, mockRequest);

      expect(result).toBeDefined();
      expect(result.text).toBe('مرحبا دكتور');
      expect(asrService.transcribe).toHaveBeenCalledWith(
        dto.audio,
        dto.callSid,
        expect.objectContaining({
          identifySpeakers: true,
          dialect: 'egypt',
        }),
      );
    });

    it('should handle transcription with diarization options', async () => {
      asrService.transcribe.mockResolvedValue(mockTranscriptionResponse);

      const dto = {
        audio: 'base64encodedaudio',
        callSid: 'CA789',
        enableDiarization: true,
        diarizeFirst: true,
      };

      const mockRequest = { headers: {} } as any;

      await controller.transcribe(dto, mockRequest);

      expect(asrService.transcribe).toHaveBeenCalledWith(
        dto.audio,
        dto.callSid,
        expect.objectContaining({
          enableDiarization: true,
          diarizeFirst: true,
        }),
      );
    });

    it('should throw error on service failure', async () => {
      asrService.transcribe.mockRejectedValue(new Error('ASR service unavailable'));

      const dto = {
        audio: 'base64encodedaudio',
        callSid: 'CA123',
      };

      const mockRequest = { headers: {} } as any;

      await expect(controller.transcribe(dto, mockRequest)).rejects.toThrow();
    });

    it('should generate callSid if not provided', async () => {
      asrService.transcribe.mockResolvedValue(mockTranscriptionResponse);

      const dto = {
        audio: 'base64encodedaudio',
      };

      const mockRequest = { headers: {} } as any;

      await controller.transcribe(dto, mockRequest);

      expect(asrService.transcribe).toHaveBeenCalledWith(
        dto.audio,
        expect.stringMatching(/^call-\d+$/),
        expect.any(Object),
      );
    });
  });

  describe('stream', () => {
    it('should stream audio transcription', async () => {
      const mockStreamResponse = { partial: 'مرحبا', final: 'مرحبا دكتور' };
      asrService.stream.mockResolvedValue(mockStreamResponse);

      const dto = {
        audio: 'base64encodedaudio',
        callSid: 'CA456',
      };

      const mockRequest = { headers: {} } as any;

      const result = await controller.stream(dto, mockRequest);

      expect(result).toEqual(mockStreamResponse);
      expect(asrService.stream).toHaveBeenCalledWith(dto.audio, dto.callSid);
    });
  });
});
