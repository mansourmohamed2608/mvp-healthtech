// gateway/src/llm/llm.controller.ts
import { Controller, Post, Body, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { LlmService, LlmResponse } from './llm.service';

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
}

@Controller('llm')
export class LlmController {
  private readonly logger = new Logger(LlmController.name);

  constructor(private readonly llmService: LlmService) {}

  @Post('infer')
  async infer(@Body() dto: InferDto): Promise<LlmResponse> {
    this.logger.log(`LLM infer request: sessionId=${dto.sessionId}`);
    try {
      const result = await this.llmService.infer(dto.message, dto.sessionId);
      return result;
    } catch (error) {
      this.logger.error('LLM infer error:', error);
      throw error;
    }
  }

  @Post('soap')
  async generateSoap(@Body() dto: SoapDto): Promise<{ soap: LlmResponse }> {
    this.logger.log('LLM SOAP generation request');
    try {
      // For now, call infer with SOAP generation prompt
      const result = await this.llmService.infer(
        `Generate a SOAP note from this transcript: ${dto.transcript}`,
        `soap-${Date.now()}`,
      );
      return { soap: result };
    } catch (error) {
      this.logger.error('LLM SOAP error:', error);
      throw error;
    }
  }

  @Post('orchestrate')
  async orchestrate(@Body() dto: OrchestrateDto) {
    this.logger.log(`Orchestrate request: sessionId=${dto.sessionId}`);
    try {
      const orchestratorUrl = process.env.ORCHESTRATOR_URL || 'http://localhost:5006';
      const response = await fetch(`${orchestratorUrl}/orchestrate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(dto),
      });

      if (!response.ok) {
        throw new Error(`Orchestrator error: ${response.statusText}`);
      }

      const result = await response.json();
      this.logger.log(`Orchestration result: intent=${result.intent}, confidence=${result.confidence}, routing=${result.routing}`);
      return result;
    } catch (error) {
      this.logger.error('Orchestration error:', error);
      throw error;
    }
  }
}
