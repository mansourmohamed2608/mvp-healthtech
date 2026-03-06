// gateway/src/asr/asr.service.ts
import { Injectable, Logger } from '@nestjs/common';
import { v4 as uuidv4 } from 'uuid';
import { InternalHttpClient } from '../http/internal-http-client.service';

/**
 * Convert raw µ-law (PCMU) bytes to a 16-bit PCM WAV buffer.
 * Twilio Media Streams send 8 kHz mono µ-law audio.
 * soundfile (Python) cannot read raw µ-law without a WAV container, so we
 * wrap it here before forwarding to the ASR /transcribe endpoint.
 */
function mulawToWav(mulawData: Buffer, sampleRate = 8000): Buffer {
  // ITU-T G.711 µ-law decode table (256 entries, pre-computed for speed)
  const ULAW_DECODE = new Int16Array(256);
  for (let i = 0; i < 256; i++) {
    let u = ~i & 0xff;
    const sign = u & 0x80;
    const exp = (u >> 4) & 0x07;
    const mantissa = u & 0x0f;
    let sample = ((mantissa | 0x10) << (exp + 3)) - 0x84;
    ULAW_DECODE[i] = sign ? -sample : sample;
  }

  const numSamples = mulawData.length;
  const pcm = Buffer.alloc(numSamples * 2);
  for (let i = 0; i < numSamples; i++) {
    pcm.writeInt16LE(ULAW_DECODE[mulawData[i]], i * 2);
  }

  // Standard 44-byte PCM WAV header
  const dataSize = pcm.length;
  const header = Buffer.alloc(44);
  header.write('RIFF', 0);
  header.writeUInt32LE(36 + dataSize, 4);
  header.write('WAVE', 8);
  header.write('fmt ', 12);
  header.writeUInt32LE(16, 16);         // fmt chunk size
  header.writeUInt16LE(1, 20);          // PCM
  header.writeUInt16LE(1, 22);          // mono
  header.writeUInt32LE(sampleRate, 24);
  header.writeUInt32LE(sampleRate * 2, 28); // byteRate
  header.writeUInt16LE(2, 32);          // blockAlign (1ch * 2B)
  header.writeUInt16LE(16, 34);         // bitsPerSample
  header.write('data', 36);
  header.writeUInt32LE(dataSize, 40);
  return Buffer.concat([header, pcm]);
}

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
  private readonly llmServiceUrl =
    process.env.LLM_SERVICE_URL || 'http://localhost:5001';
  private readonly internalSecret = (() => {
    if (!process.env.INTERNAL_SECRET)
      throw new Error('INTERNAL_SECRET not set');
    return process.env.INTERNAL_SECRET;
  })();
  constructor(private readonly http: InternalHttpClient) {}

  // Batch transcription with optional speaker role detection
  async transcribe(
    audio: string,
    callSid: string,
    opts?: {
      identifySpeakers?: boolean;
      dialect?: string;
      language?: string;
      enableDiarization?: boolean;
      diarizeFirst?: boolean;
      enableAlignment?: boolean;
      format?: string;   // 'mulaw' | 'wav' — default wav/soundfile-compatible
      sampleRate?: number;
    },
    correlationId?: string,
  ): Promise<TranscriptionResponse> {
    const corr = correlationId || uuidv4();
    const client = this.http.getClient({
      baseUrl: process.env.ASR_SERVICE_URL || '',
      serviceName: 'asr',
    });

    // Twilio sends raw µ-law audio which soundfile cannot decode directly.
    // Convert to a 16-bit PCM WAV before forwarding to the ASR service.
    let audioToSend = audio;
    if (opts?.format === 'mulaw') {
      const mulawBytes = Buffer.from(audio, 'base64');
      const wavBytes = mulawToWav(mulawBytes, opts?.sampleRate ?? 8000);
      audioToSend = wavBytes.toString('base64');
    }

    const identifySpeakers = opts?.identifySpeakers ?? true;
    const { data } = await client.post<TranscriptionResponse>(
      `/transcribe`,
      {
        audio: audioToSend,
        callSid,
        dialect: opts?.dialect,
        language: opts?.language,
        enable_diarization: opts?.enableDiarization,
        diarize_first: opts?.diarizeFirst,
        enable_alignment: opts?.enableAlignment,
      },
      { headers: { 'x-correlation-id': corr } },
    );

    // If speaker diarization was enabled and we have segments, identify roles
    if (identifySpeakers && data.segments && data.segments.length > 0) {
      try {
        this.logger.log(
          `Identifying speaker roles for ${data.segments.length} segments`,
        );
        const llmClient = this.http.getClient({
          baseUrl: this.llmServiceUrl,
          serviceName: 'llm',
        });
        const roleResponse = await llmClient.post(`/identify-speakers`, {
          segments: data.segments,
          context: 'medical',
        });

        // Add role information to the response
        data.roles = roleResponse.data.roles;
        data.primary_doctor = roleResponse.data.primary_doctor;
        data.primary_patient = roleResponse.data.primary_patient;

        this.logger.log(`Roles identified for call ${callSid}`);
      } catch (error) {
        this.logger.warn(`Failed to identify speaker roles: ${error.message}`);
        // Continue without role detection - not critical
      }
    }

    return data;
  }

  // Streaming transcription
  async stream(
    audio: string,
    callSid: string,
    isFinal = false,
  ): Promise<StreamResponse> {
    const corr = uuidv4();
    const client = this.http.getClient({
      baseUrl: process.env.ASR_SERVICE_URL || '',
      serviceName: 'asr',
    });
    const { data } = await client.post<StreamResponse>(
      `/stream/chunk`,
      { audio, sessionId: callSid, isFinal },
      { headers: { 'x-correlation-id': corr } },
    );
    return data;
  }
}
