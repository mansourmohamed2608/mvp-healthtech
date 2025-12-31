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

export interface OrchestratePayload {
  transcript: string;
  sessionId: string;
  mode?: string;
  history?: ChatMessage[];
  slots?: Record<string, any>;
  dialect?: string;
}

export interface OrchestrateResponse extends LlmResponse {
  slots?: Record<string, any>;
}

@Injectable()
export class LlmService {
  private readonly logger = new Logger(LlmService.name);
  private readonly internalSecret = (() => {
    if (!process.env.INTERNAL_SECRET) throw new Error('INTERNAL_SECRET not set');
    return process.env.INTERNAL_SECRET;
  })();
  private readonly serviceUrl = process.env.LLM_SERVICE_URL || '';
  private readonly orchestratorUrl = process.env.ORCHESTRATOR_URL || '';

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

  async orchestrate(payload: OrchestratePayload): Promise<OrchestrateResponse> {
    const corr = uuidv4();
    const timeoutMs = payload.mode === 'voice_agent_va' ? 2000 : 20000;
    const client = this.http.getClient({
      baseUrl: this.orchestratorUrl || this.serviceUrl,
      serviceName: 'orchestrator',
      timeoutMs,
    });
    const { data } = await client.post<OrchestrateResponse>(
      `/orchestrate`,
      payload,
      {
        headers: {
          'x-correlation-id': corr,
          'x-internal-secret': this.internalSecret,
        },
        timeout: timeoutMs,
      },
    );
    return data;
  }
}
