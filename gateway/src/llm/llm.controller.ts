// gateway/src/llm/llm.controller.ts
import { Controller, Post, Body, Logger, UseGuards, Req } from '@nestjs/common';
import { LlmService, LlmResponse } from './llm.service';
import { JwtAuthGuard } from '../auth/jwt.guard';
import { TenantGuard } from '../auth/tenant.guard';
import { wrapError, camelResponse } from '../utils/http-utils';
import type { Request } from 'express';
import { MetricsController } from '../metrics/metrics.controller';

class InferDto {
  message: string;
  sessionId: string;
}

class SoapDto {
  transcript: string;
}

class OrchestrateDto {
  transcript: string;
  sessionId: string;
  context?: Record<string, any>;
  tenantId?: string;
}

class ChatDto {
  message: string;
  history?: { role: string; content: string }[];
  sessionId: string;
  intent?: string;
}

@UseGuards(JwtAuthGuard, TenantGuard)
@Controller('llm')
export class LlmController {
  private readonly logger = new Logger(LlmController.name);
  private readonly llmLatency = MetricsController.getLlmLatency();

  constructor(private readonly llmService: LlmService) {}

  @Post('infer')
  async infer(
    @Body() dto: InferDto,
    @Req() req: Request,
  ): Promise<LlmResponse> {
    this.logger.log(`LLM infer request: sessionId=${dto.sessionId}`);
    const start = process.hrtime();
    try {
      const result = await this.llmService.infer(dto.message, dto.sessionId);
      this.llmLatency.observe(
        { endpoint: 'infer', status: 'ok' },
        this.durationSeconds(start),
      );
      return camelResponse(result);
    } catch (error) {
      this.llmLatency.observe(
        { endpoint: 'infer', status: 'error' },
        this.durationSeconds(start),
      );
      wrapError(error, req);
    }
  }

  @Post('soap')
  async generateSoap(@Body() dto: SoapDto): Promise<{ soap: LlmResponse }> {
    this.logger.log('LLM SOAP generation request');
    const start = process.hrtime();
    try {
      // For now, call infer with SOAP generation prompt
      const result = await this.llmService.infer(
        `Generate a SOAP note from this transcript: ${dto.transcript}`,
        `soap-${Date.now()}`,
      );
      this.llmLatency.observe(
        { endpoint: 'soap', status: 'ok' },
        this.durationSeconds(start),
      );
      return { soap: result };
    } catch (error) {
      this.llmLatency.observe(
        { endpoint: 'soap', status: 'error' },
        this.durationSeconds(start),
      );
      wrapError(error);
    }
  }

  @Post('chat')
  async chat(@Body() dto: ChatDto, @Req() req: Request): Promise<LlmResponse> {
    this.logger.log(`LLM chat request: sessionId=${dto.sessionId}`);
    const start = process.hrtime();
    try {
      const result = await this.llmService.chat({
        message: dto.message,
        history: dto.history,
        sessionId: dto.sessionId,
        intent: dto.intent,
      });
      this.llmLatency.observe(
        { endpoint: 'chat', status: 'ok' },
        this.durationSeconds(start),
      );
      return camelResponse(result);
    } catch (error) {
      this.llmLatency.observe(
        { endpoint: 'chat', status: 'error' },
        this.durationSeconds(start),
      );
      wrapError(error, req);
    }
  }

  @Post('orchestrate')
  async orchestrate(@Body() dto: OrchestrateDto) {
    this.logger.log(`Orchestrate request: sessionId=${dto.sessionId}`);
    try {
      const orchestratorUrl =
        process.env.ORCHESTRATOR_URL || 'http://localhost:5006';
      const response = await fetch(`${orchestratorUrl}/orchestrate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(process.env.INTERNAL_SECRET
            ? { 'x-internal-secret': process.env.INTERNAL_SECRET }
            : {}),
        },
        body: JSON.stringify(dto),
      });

      if (!response.ok) {
        throw new Error(`Orchestrator error: ${response.statusText}`);
      }

      const result = await response.json();
      this.logger.log(
        `Orchestration result: intent=${result.intent}, confidence=${result.confidence}, routing=${result.routing}`,
      );
      return result;
    } catch (error) {
      wrapError(error);
    }
  }

  private durationSeconds(start: [number, number]) {
    const diff = process.hrtime(start);
    return diff[0] + diff[1] / 1e9;
  }
}
