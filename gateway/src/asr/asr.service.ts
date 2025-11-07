// gateway/src/asr/asr.service.ts
import { Injectable, Logger } from '@nestjs/common';
import axios from 'axios';

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

  // Batch transcription with optional speaker role detection
  async transcribe(
    audio: string,
    callSid: string,
    identifySpeakers: boolean = true
  ): Promise<TranscriptionResponse> {
    const { data } = await axios.post<TranscriptionResponse>(
      `${process.env.ASR_SERVICE_URL}/transcribe`,
      { audio, callSid },
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
          }
        );

        // Add role information to the response
        data.roles = roleResponse.data.roles;
        data.primary_doctor = roleResponse.data.primary_doctor;
        data.primary_patient = roleResponse.data.primary_patient;

        this.logger.log(
          `Roles identified: Doctor=${data.primary_doctor}, Patient=${data.primary_patient}`
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
    const { data } = await axios.post<StreamResponse>(
      `${process.env.ASR_SERVICE_URL}/stream`,
      { audio, callSid },
    );
    return data;
  }
}
