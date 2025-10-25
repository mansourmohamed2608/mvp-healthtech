// gateway/src/llm/llm.service.ts
import { Injectable, Logger } from '@nestjs/common';
import axios from 'axios';

interface LlmResponse {
  reply: string;
  intent?: string;
  confidence?: number;
  [key: string]: unknown;
}

@Injectable()
export class LlmService {
  private readonly logger = new Logger(LlmService.name);
  
  async infer(message: string, sessionId: string): Promise<LlmResponse> {
    const { data } = await axios.post<LlmResponse>(
      `${process.env.LLM_SERVICE_URL}/infer`,
      { message, sessionId },
    );
    return data;
  }
}
