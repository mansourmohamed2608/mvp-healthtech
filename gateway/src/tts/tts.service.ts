// gateway/src/tts/tts.service.ts
/**
 * TTS Service Client
 * Communicates with TTS microservice for speech synthesis
 * Week 3 Day 16 (Oct 10, 2025)
 */
import * as http from 'http';
import { Injectable, Logger } from '@nestjs/common';
import { v4 as uuidv4 } from 'uuid';
import { InternalHttpClient } from '../http/internal-http-client.service';

interface SynthesizeRequest {
  text: string;
  voice?: string;
  sessionId?: string;
  format?: 'wav' | 'mp3' | 'mulaw';
}

interface SynthesizeResponse {
  audio: string; // Base64 encoded
  duration: number;
  sampleRate: number;
  format?: string;
}

@Injectable()
export class TtsService {
  private readonly logger = new Logger(TtsService.name);
  private readonly serviceUrl =
    process.env.TTS_SERVICE_URL || 'http://localhost:5002';
  private readonly internalSecret = (() => {
    if (!process.env.INTERNAL_SECRET)
      throw new Error('INTERNAL_SECRET not set');
    return process.env.INTERNAL_SECRET;
  })();
  constructor(private readonly http: InternalHttpClient) {}

  /**
   * Synthesize speech from text
   * Uses native node:http to avoid Axios retry body-serialization bug (Axios 1.x)
   */
  async synthesize(
    text: string,
    sessionId?: string,
    voice?: string,
  ): Promise<{ audioBase64: string; format?: string }> {
    const corr = uuidv4();
    const body = JSON.stringify({ text, sessionId, voice });
    const url = new URL(this.serviceUrl);
    const hostname = url.hostname;
    const port = parseInt(url.port || '5002', 10);

    return new Promise((resolve, reject) => {
      const req = http.request(
        {
          hostname,
          port,
          path: '/synthesize',
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Content-Length': Buffer.byteLength(body),
            'x-internal-secret': this.internalSecret,
            'x-correlation-id': corr,
          },
          timeout: 30000,
        },
        (res) => {
          let raw = '';
          res.on('data', (chunk) => (raw += chunk));
          res.on('end', () => {
            if (res.statusCode === 200) {
              const j = JSON.parse(raw);
              this.logger.log(`Synthesized ${text.length} chars`);
              resolve({ audioBase64: j.audio, format: j.format || 'mulaw' });
            } else {
              this.logger.error(
                `TTS synthesis failed: ${res.statusCode} ${raw.substring(0, 200)}`,
              );
              reject(new Error(`TTS ${res.statusCode}`));
            }
          });
        },
      );
      req.on('error', (e) => {
        this.logger.error(`TTS request error: ${e.message}`);
        reject(e);
      });
      req.on('timeout', () => {
        req.destroy();
        reject(new Error('TTS request timeout'));
      });
      req.write(body);
      req.end();
    });
  }

  /**
   * Synthesize and stream audio chunks
   * For real-time playback
   */
  async synthesizeStream(
    text: string,
    sessionId?: string,
    voice?: string,
  ): Promise<Buffer> {
    try {
      const client = this.http.getClient({
        baseUrl: this.serviceUrl,
        serviceName: 'tts',
      });
      const response = await client.post(
        `/synthesize/stream`,
        {
          text,
          sessionId,
          voice,
        } as SynthesizeRequest,
        {
          responseType: 'arraybuffer',
          timeout: 15000, // 15s timeout for streaming
          headers: { 'x-correlation-id': sessionId || uuidv4() },
        },
      );

      return Buffer.from(response.data);
    } catch (error) {
      this.logger.error(`TTS stream synthesis failed: ${error}`);
      throw error;
    }
  }

  /**
   * Get list of available voices
   */
  async getVoices(): Promise<any> {
    try {
      const client = this.http.getClient({
        baseUrl: this.serviceUrl,
        serviceName: 'tts',
      });
      const response = await client.get(`/voices`);
      return response.data;
    } catch (error) {
      this.logger.error(`Failed to fetch voices: ${error}`);
      return { voices: ['egtts', 'saudi-tts', 'ar-EG-SalmaNeural'] }; // Fallback
    }
  }
}
