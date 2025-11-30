// gateway/src/llm/llm.service.ts
import { Injectable, Logger } from '@nestjs/common';
import { v4 as uuidv4 } from 'uuid';
import { InternalHttpClient } from '../http/internal-http-client.service';

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
  private readonly serviceUrl = process.env.LLM_SERVICE_URL || '';

  constructor(private readonly http: InternalHttpClient) {}

  async infer(message: string, sessionId: string): Promise<LlmResponse> {
    const corr = uuidv4();
    const client = this.http.getClient({ baseUrl: this.serviceUrl, serviceName: 'llm' });
    const { data } = await client.post<LlmResponse>(
      `/infer`,
      { message, sessionId },
      { headers: { 'x-correlation-id': corr } },
    );
    return data;
  }

  async chat(payload: { message: string; history?: ChatMessage[]; sessionId: string; intent?: string }): Promise<LlmResponse> {
    const corr = uuidv4();
    const client = this.http.getClient({ baseUrl: this.serviceUrl, serviceName: 'llm' });
    const { data } = await client.post<LlmResponse>(
      `/chat`,
      payload,
      { headers: { 'x-correlation-id': corr } },
    );
    return data;
  }
}
