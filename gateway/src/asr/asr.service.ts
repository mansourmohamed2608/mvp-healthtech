// gateway/src/asr/asr.service.ts
import { Injectable, Logger } from '@nestjs/common';
import axios from 'axios';
import { v4 as uuidv4 } from 'uuid';

// Response types for ASR service
export interface SpeakerSegment {
  speaker: string;
  text: string;
  start: number;
  end: number;
}

export interface SpeakerRole {
  speaker_id: string;
  role: string;
  confidence: number;
  reasoning: string;
}

export interface TranscriptionResponse {
  text: string;
  confidence?: number;
  language?: string;
  segments?: SpeakerSegment[];
  speakers?: string[];
  roles?: SpeakerRole[];
  primary_doctor?: string;
  primary_patient?: string;
  [key: string]: unknown;
}

export interface StreamResponse {
  partial?: string;
  final?: string;
  [key: string]: unknown;
}

@Injectable()
export class AsrService {
  private readonly logger = new Logger(AsrService.name);
  private readonly llmServiceUrl = process.env.LLM_SERVICE_URL || 'http://localhost:5001';
  private readonly internalSecret = (() => {
    if (!process.env.INTERNAL_SECRET) throw new Error('INTERNAL_SECRET not set');
    return process.env.INTERNAL_SECRET;
  })();

  // Batch transcription with optional speaker role detection
  async transcribe(
    audio: string,
    callSid: string,
    identifySpeakers: boolean = true,
    correlationId?: string,
  ): Promise<TranscriptionResponse> {
    const corr = correlationId || uuidv4();
    const { data } = await axios.post<TranscriptionResponse>(
      `${process.env.ASR_SERVICE_URL}/transcribe`,
      { audio, callSid },
      { timeout: 15000, headers: { 'x-correlation-id': corr, 'x-internal-secret': this.internalSecret } },
    );

    // If speaker diarization was enabled and we have segments, identify roles
    if (identifySpeakers && data.segments && data.segments.length > 0) {
      try {
        this.logger.log(`Identifying speaker roles for ${data.segments.length} segments`);
        const roleResponse = await axios.post(
          `${this.llmServiceUrl}/identify-speakers`,
          {
            segments: data.segments,
            context: 'medical'
          },
          { headers: { 'x-internal-secret': this.internalSecret } },
        );

        // Add role information to the response
        data.roles = roleResponse.data.roles;
        data.primary_doctor = roleResponse.data.primary_doctor;
        data.primary_patient = roleResponse.data.primary_patient;

        this.logger.log(
          `Roles identified for call ${callSid}`,
        );
      } catch (error) {
        this.logger.warn(`Failed to identify speaker roles: ${error.message}`);
        // Continue without role detection - not critical
      }
    }

    return data;
  }

  // Streaming transcription
  async stream(audio: string, callSid: string): Promise<StreamResponse> {
    const corr = uuidv4();
    // Reuse the same /transcribe endpoint to keep one canonical contract
    const { data } = await axios.post<TranscriptionResponse>(
      `${process.env.ASR_SERVICE_URL}/transcribe`,
      { audio, callSid },
      { timeout: 15000, headers: { 'x-correlation-id': corr, 'x-internal-secret': this.internalSecret } },
    );
    return {
      partial: data.text,
      final: data.text,
      ...data,
    };
  }
}
