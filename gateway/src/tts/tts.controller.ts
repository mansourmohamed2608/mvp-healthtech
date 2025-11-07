// gateway/src/tts/tts.controller.ts
import { Controller, Post, Body, Logger } from '@nestjs/common';
import axios from 'axios';

class SynthesizeDto {
  text: string;
  voice?: string;
}

@Controller('tts')
export class TtsController {
  private readonly logger = new Logger(TtsController.name);
  private readonly ttsServiceUrl = process.env.TTS_SERVICE_URL || 'http://localhost:5002';

  @Post('synthesize')
  async synthesize(@Body() dto: SynthesizeDto) {
    this.logger.log('TTS synthesize request');
    try {
      const response = await axios.post(`${this.ttsServiceUrl}/synthesize`, {
        text: dto.text,
        voice: dto.voice,
      });
      return response.data;
    } catch (error) {
      this.logger.error('TTS synthesize error:', error);
      throw error;
    }
  }
}
