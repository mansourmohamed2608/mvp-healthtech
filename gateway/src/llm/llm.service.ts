// gateway/src/llm/llm.service.ts
import { Injectable, Logger } from '@nestjs/common';
import axios from 'axios';
import { v4 as uuidv4 } from 'uuid';

export interface LlmResponse {
  reply: string;
  intent?: string;
  confidence?: number;
  [key: string]: unknown;
}

export interface ChatMessage {
  role: string;
  content: string;
}

@Injectable()
export class LlmService {
  private readonly logger = new Logger(LlmService.name);
  private readonly internalSecret = (() => {
    if (!process.env.INTERNAL_SECRET) throw new Error('INTERNAL_SECRET not set');
    return process.env.INTERNAL_SECRET;
  })();

  async infer(message: string, sessionId: string): Promise<LlmResponse> {
    const corr = uuidv4();
    const { data } = await axios.post<LlmResponse>(
      `${process.env.LLM_SERVICE_URL}/infer`,
      { message, sessionId },
      { timeout: 15000, headers: { 'x-correlation-id': corr, 'x-internal-secret': this.internalSecret } },
    );
    return data;
  }

  async chat(payload: { message: string; history?: ChatMessage[]; sessionId: string; intent?: string }): Promise<LlmResponse> {
    const corr = uuidv4();
    const { data } = await axios.post<LlmResponse>(
      `${process.env.LLM_SERVICE_URL}/chat`,
      payload,
      { timeout: 15000, headers: { 'x-correlation-id': corr, 'x-internal-secret': this.internalSecret } },
    );
    return data;
  }
}
