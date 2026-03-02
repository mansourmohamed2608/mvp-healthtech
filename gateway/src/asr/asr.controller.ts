// gateway/src/asr/asr.controller.ts
import { Controller, Post, Body, Logger, UseGuards, Req } from '@nestjs/common';
import { Throttle } from '@nestjs/throttler';
import {
  AsrService,
  TranscriptionResponse,
  StreamResponse,
} from './asr.service';
import { JwtAuthGuard } from '../auth/jwt.guard';
import { TenantGuard } from '../auth/tenant.guard';
import { wrapError, camelResponse } from '../utils/http-utils';
import type { Request } from 'express';
import { MetricsController } from '../metrics/metrics.controller';

class TranscribeDto {
  audio: string;
  callSid?: string;
  dialect?: string;
  language?: string;
  enableDiarization?: boolean;
  diarizeFirst?: boolean;
}

class StreamDto {
  audio: string;
  callSid: string;
  dialect?: string;
}

@Throttle({ gpu: { ttl: 60_000, limit: 5 } }) // 5 GPU transcriptions per IP per minute
@UseGuards(JwtAuthGuard, TenantGuard)
@Controller('asr')
export class AsrController {
  private readonly logger = new Logger(AsrController.name);
  private readonly asrLatency = MetricsController.getAsrLatency();

  constructor(private readonly asrService: AsrService) {}

  @Post('transcribe')
  async transcribe(
    @Body() dto: TranscribeDto,
    @Req() req: Request,
  ): Promise<TranscriptionResponse> {
    this.logger.log(
      `Transcribe request: callSid=${dto.callSid}, dialect=${dto.dialect}`,
    );
    const start = process.hrtime();
    try {
      // Pass true to enable speaker role identification (default)
      const result = await this.asrService.transcribe(
        dto.audio,
        dto.callSid || `call-${Date.now()}`,
        {
          identifySpeakers: true,
          dialect: dto.dialect,
          language: dto.language,
          enableDiarization: dto.enableDiarization,
          diarizeFirst: dto.diarizeFirst,
        },
      );
      this.asrLatency.observe(
        { endpoint: 'transcribe', status: 'ok' },
        this.durationSeconds(start),
      );
      return camelResponse(result);
    } catch (error) {
      this.asrLatency.observe(
        { endpoint: 'transcribe', status: 'error' },
        this.durationSeconds(start),
      );
      wrapError(error, req);
    }
  }

  @Post('stream')
  async stream(
    @Body() dto: StreamDto,
    @Req() req: Request,
  ): Promise<StreamResponse> {
    this.logger.log(
      `Stream request (legacy alias of /transcribe): callSid=${dto.callSid}`,
    );
    const start = process.hrtime();
    try {
      const result = await this.asrService.stream(dto.audio, dto.callSid);
      this.asrLatency.observe(
        { endpoint: 'stream', status: 'ok' },
        this.durationSeconds(start),
      );
      return camelResponse(result);
    } catch (error) {
      this.asrLatency.observe(
        { endpoint: 'stream', status: 'error' },
        this.durationSeconds(start),
      );
      wrapError(error, req);
    }
  }

  private durationSeconds(start: [number, number]) {
    const diff = process.hrtime(start);
    return diff[0] + diff[1] / 1e9;
  }
}
