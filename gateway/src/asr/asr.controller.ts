// gateway/src/asr/asr.controller.ts
import { Controller, Post, Body, Logger } from '@nestjs/common';
import { AsrService, TranscriptionResponse, StreamResponse } from './asr.service';

class TranscribeDto {
  audio: string;
  callSid?: string;
  dialect?: string;
}

class StreamDto {
  audio: string;
  callSid: string;
  dialect?: string;
}

@Controller('asr')
export class AsrController {
  private readonly logger = new Logger(AsrController.name);

  constructor(private readonly asrService: AsrService) {}

  @Post('transcribe')
  async transcribe(@Body() dto: TranscribeDto): Promise<TranscriptionResponse> {
    this.logger.log(`Transcribe request: callSid=${dto.callSid}, dialect=${dto.dialect}`);
    try {
      // Pass true to enable speaker role identification (default)
      const result = await this.asrService.transcribe(
        dto.audio,
        dto.callSid || `call-${Date.now()}`,
        true  // Enable speaker role detection
      );
      return result;
    } catch (error) {
      this.logger.error('ASR transcribe error:', error);
      throw error;
    }
  }

  @Post('stream')
  async stream(@Body() dto: StreamDto): Promise<StreamResponse> {
    this.logger.log(`Stream request: callSid=${dto.callSid}`);
    try {
      const result = await this.asrService.stream(dto.audio, dto.callSid);
      return result;
    } catch (error) {
      this.logger.error('ASR stream error:', error);
      throw error;
    }
  }
}
