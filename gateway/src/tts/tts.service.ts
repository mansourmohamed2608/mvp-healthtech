// gateway/src/tts/tts.service.ts
/**
 * TTS Service Client
 * Communicates with TTS microservice for speech synthesis
 * Week 3 Day 16 (Oct 10, 2025)
 */
import { Injectable, Logger } from '@nestjs/common';
import axios from 'axios';
import { v4 as uuidv4 } from 'uuid';

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
  private readonly serviceUrl = process.env.TTS_SERVICE_URL || 'http://localhost:5002';
  private readonly internalSecret = (() => {
    if (!process.env.INTERNAL_SECRET) throw new Error('INTERNAL_SECRET not set');
    return process.env.INTERNAL_SECRET;
  })();

  /**
   * Synthesize speech from text
   * Returns base64 audio (mulaw/mp3/wav depending on engine)
   */
  async synthesize(text: string, sessionId?: string): Promise<{ audioBase64: string; format?: string }> {
    try {
      const corr = uuidv4();
      const response = await axios.post(
        `${this.serviceUrl}/synthesize`,
        {
          text,
          sessionId,
          voice: 'ar-EG-SalmaNeural', // Arabic Egyptian female
        } as SynthesizeRequest,
        {
          timeout: 15000,
          headers: { 'x-correlation-id': corr, 'x-internal-secret': this.internalSecret },
        },
      );

      this.logger.log(`Synthesized ${text.length} chars`);
      return { audioBase64: response.data.audio, format: response.data.format || 'mulaw' };
    } catch (error) {
      this.logger.error(`TTS synthesis failed: ${error}`);
      throw error;
    }
  }

  /**
   * Synthesize and stream audio chunks
   * For real-time playback
   */
  async synthesizeStream(text: string, sessionId?: string): Promise<Buffer> {
    try {
      const response = await axios.post(
        `${this.serviceUrl}/synthesize/stream`,
        {
          text,
          sessionId,
          voice: 'ar-EG-SalmaNeural',
        } as SynthesizeRequest,
        {
          responseType: 'arraybuffer',
          timeout: 15000, // 15s timeout for streaming
          headers: { 'x-internal-secret': this.internalSecret },
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
      const response = await axios.get(`${this.serviceUrl}/voices`, { headers: { 'x-internal-secret': this.internalSecret } });
      return response.data;
    } catch (error) {
      this.logger.error(`Failed to fetch voices: ${error}`);
      return { voices: ['ar-EG-SalmaNeural'] }; // Fallback
    }
  }
}
