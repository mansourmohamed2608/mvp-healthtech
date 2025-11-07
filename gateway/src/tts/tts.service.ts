// gateway/src/tts/tts.service.ts
/**
 * TTS Service Client
 * Communicates with TTS microservice for speech synthesis
 * Week 3 Day 16 (Oct 10, 2025)
 */
import { Injectable, Logger } from '@nestjs/common';
import axios from 'axios';

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
}

@Injectable()
export class TtsService {
  private readonly logger = new Logger(TtsService.name);
  private readonly serviceUrl = process.env.TTS_SERVICE_URL || 'http://localhost:5002';

  /**
   * Synthesize speech from text
   * Returns audio stream
   */
  async synthesize(text: string, sessionId?: string): Promise<Buffer> {
    try {
      const response = await axios.post(
        `${this.serviceUrl}/synthesize`,
        {
          text,
          sessionId,
          voice: 'ar-EG-SalmaNeural', // Arabic Egyptian female
        } as SynthesizeRequest,
        {
          responseType: 'arraybuffer',
          timeout: 10000, // 10s timeout
        },
      );

      this.logger.log(`Synthesized ${text.length} chars in ${response.headers['x-duration']}s`);
      return Buffer.from(response.data);
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
      const response = await axios.get(`${this.serviceUrl}/voices`);
      return response.data;
    } catch (error) {
      this.logger.error(`Failed to fetch voices: ${error}`);
      return { voices: ['ar-EG-SalmaNeural'] }; // Fallback
    }
  }
}
