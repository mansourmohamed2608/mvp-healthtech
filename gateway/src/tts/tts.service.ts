// gateway/src/tts/tts.service.ts
/**
 * TTS Service Client
 * Communicates with TTS microservice for speech synthesis
 * Week 3 Day 16 (Oct 10, 2025)
 */
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
   * Returns base64 audio (mulaw/mp3/wav depending on engine)
   */
  async synthesize(
    text: string,
    sessionId?: string,
    voice?: string,
  ): Promise<{ audioBase64: string; format?: string }> {
    try {
      const corr = uuidv4();
      const client = this.http.getClient({
        baseUrl: this.serviceUrl,
        serviceName: 'tts',
        timeoutMs: 30000, // XTTS Arabic GPU inference ~1-2s; 30s safety margin
      });
      const response = await client.post(
        `/synthesize`,
        {
          text,
          sessionId,
          voice,
        } as SynthesizeRequest,
        {
          headers: { 'x-correlation-id': corr },
        },
      );

      this.logger.log(`Synthesized ${text.length} chars`);
      return {
        audioBase64: response.data.audio,
        format: response.data.format || 'mulaw',
      };
    } catch (error) {
      this.logger.error(`TTS synthesis failed: ${error}`);
      throw error;
    }
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
