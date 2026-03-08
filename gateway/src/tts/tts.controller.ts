// gateway/src/tts/tts.controller.ts
import { Controller, Post, Body, Logger, UseGuards, Req } from '@nestjs/common';
import { IsString, IsOptional } from 'class-validator';
import { JwtAuthGuard } from '../auth/jwt.guard';
import { TenantGuard } from '../auth/tenant.guard';
import { TtsService } from './tts.service';
import { wrapError, camelResponse } from '../utils/http-utils';
import { Request } from 'express';
import { MetricsController } from '../metrics/metrics.controller';

class SynthesizeDto {
  @IsString()
  text: string;

  @IsOptional()
  @IsString()
  voice?: string;
}

@UseGuards(JwtAuthGuard, TenantGuard)
@Controller('tts')
export class TtsController {
  private readonly logger = new Logger(TtsController.name);
  private readonly ttsLatency = MetricsController.getTtsLatency();
  constructor(private readonly ttsService: TtsService) {}

  @Post('synthesize')
  async synthesize(@Body() dto: SynthesizeDto, @Req() req: any) {
    this.logger.log('TTS synthesize request');
    const start = process.hrtime();
    try {
      const result = await this.ttsService.synthesize(dto.text, undefined, dto.voice);
      // Normalize response shape for clients (base64 audio payload)
      this.ttsLatency.observe(
        { endpoint: 'synthesize', status: 'ok' },
        this.durationSeconds(start),
      );
      return {
        audio: result.audioBase64,
        format: result.format || 'mulaw',
      };
    } catch (error) {
      this.ttsLatency.observe(
        { endpoint: 'synthesize', status: 'error' },
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
