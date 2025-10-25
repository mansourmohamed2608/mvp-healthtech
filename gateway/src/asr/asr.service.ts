// gateway/src/asr/asr.service.ts
import { Injectable, Logger } from '@nestjs/common';
import axios from 'axios';

// Response types for ASR service
interface TranscriptionResponse {
  text: string;
  confidence?: number;
  language?: string;
  [key: string]: unknown;
}

interface StreamResponse {
  partial?: string;
  final?: string;
  [key: string]: unknown;
}

@Injectable()
export class AsrService {
  private readonly logger = new Logger(AsrService.name);

  // Batch transcription
  async transcribe(audio: string, callSid: string): Promise<TranscriptionResponse> {
    const { data } = await axios.post<TranscriptionResponse>(
      `${process.env.ASR_SERVICE_URL}/transcribe`,
      { audio, callSid },
    );
    return data;
  }

  // Streaming transcription
  async stream(audio: string, callSid: string): Promise<StreamResponse> {
    const { data } = await axios.post<StreamResponse>(
      `${process.env.ASR_SERVICE_URL}/stream`,
      { audio, callSid },
    );
    return data;
  }
}
