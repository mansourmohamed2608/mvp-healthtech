// gateway/src/voice/voice.gateway.ts
import {
  WebSocketGateway,
  WebSocketServer,
  OnGatewayConnection,
  OnGatewayDisconnect,
  SubscribeMessage,
  MessageBody,
  ConnectedSocket,
} from '@nestjs/websockets';
import { Logger, UseGuards } from '@nestjs/common';
import { Server, WebSocket } from 'ws';
import { ConversationService } from '../conversation/conversation.service';
import { SessionService } from '../session/session.service';
import { WsJwtGuard } from '../auth/ws-jwt.guard';
import { MetricsController } from '../metrics/metrics.controller';
import { safeLog } from '../utils/safe-logger';
import { AsrService } from '../asr/asr.service';

interface TwilioStartPayload {
  streamSid: string;
  accountSid: string;
  callSid: string;
  tracks: string[];
  mediaFormat: {
    encoding: string;
    sampleRate: number;
    channels: number;
  };
  customParameters?: Record<string, string>;
}

interface TwilioMediaMessage {
  event: 'connected' | 'start' | 'media' | 'stop' | 'mark';
  streamSid?: string;
  start?: TwilioStartPayload;
  media?: {
    track: string;
    chunk: string;
    timestamp: string;
    payload: string; // base64 audio
  };
  mark?: {
    name: string;
  };
  stop?: {
    accountSid: string;
    callSid: string;
  };
}

// WS auth: JWT for browser clients, HMAC via Custom Parameters for Twilio streams
@UseGuards(WsJwtGuard)
@WebSocketGateway({
  // Twilio connects to /twilio/{callSid} - we accept all WS connections
  // and filter by path in handleConnection
  cors: {
    origin: '*',
  },
})
export class VoiceGateway implements OnGatewayConnection, OnGatewayDisconnect {
  @WebSocketServer()
  server: Server;

  private readonly logger = new Logger(VoiceGateway.name);
  private readonly activeStreams = new Map<string, WebSocket>();
  private readonly audioBuffers = new Map<string, Buffer[]>();
  private readonly streamUsers = new Map<string, any>();
  private readonly rateLimit = new Map<string, { count: number; ts: number }>(); // per-callSid per second
  private readonly streamSids = new Map<string, string>(); // callSid -> streamSid
  private readonly activeGauge = MetricsController.getActiveConversations();
  /** Tracks calls whose pipeline (ASR→LLM→TTS) is currently running. */
  private readonly processingCalls = new Set<string>();
  /** Debounce timers for silence-gap utterance detection (one per active call). */
  private readonly silenceTimers = new Map<string, ReturnType<typeof setTimeout>>();

  // VAD constants
  private static readonly MIN_BUFFER_BYTES = 8000;  // 1s minimum before trigger
  private static readonly MAX_BUFFER_BYTES = 192000; // 24s safety cap
  private static readonly SILENCE_GAP_MS = 800;     // 800ms silence = end of utterance

  constructor(
    private readonly conversationService: ConversationService,
    private readonly sessionService: SessionService,
    private readonly asrService: AsrService,
  ) {}

  handleConnection(client: WebSocket, request: any) {
    const url = request?.url || '';
    const urlCallSid = this.extractCallSidFromUrl(url);
    
    // Only accept connections to /twilio/* paths
    if (!url.startsWith('/twilio/') && !url.startsWith('/twilio?')) {
      this.logger.debug(`Rejecting non-Twilio WebSocket path: ${url}`);
      client.close(1008, 'Invalid path');
      return;
    }
    
    safeLog(this.logger, 'log', 'WebSocket connected (pending auth)', {
      callSid: urlCallSid || 'unknown',
      url: url,
    });

    // For Twilio streams, auth is validated on 'start' message, not connection
    // Mark client as pending - will be validated when 'start' event arrives
    const user = (client as any).user || {};
    const pendingTwilioAuth = (client as any).pendingTwilioAuth;

    // WsJwtGuard runs on @SubscribeMessage handlers, NOT on handleConnection.
    // Detect Twilio media streams by CallSid pattern (Twilio CallSids start with 'CA').
    const isTwilioStream = !!urlCallSid?.startsWith('CA');

    if (pendingTwilioAuth || isTwilioStream) {
      // Twilio stream - auth will be validated on 'start' message
      (client as any).pendingTwilioAuth = true;
      (client as any).user =
        (client as any).user || { sub: 'twilio_pending', roles: ['twilio'] };
      safeLog(
        this.logger,
        'log',
        'Twilio stream connection pending auth validation',
        { callSid: urlCallSid },
      );
      return;
    }

    // Non-Twilio client (browser) - require JWT auth
    if (!user || !user.sub) {
      this.logger.warn('Unauthorized non-Twilio WS connection attempt');
      client.close();
      return;
    }

    // JWT-authenticated browser client
    if (urlCallSid) {
      this.activeStreams.set(urlCallSid, client);
      this.audioBuffers.set(urlCallSid, []);
      this.streamUsers.set(urlCallSid, user);
      this.activeGauge.inc();
    }
  }

  handleDisconnect(client: WebSocket) {
    // Find and remove the disconnected client
    for (const [callSid, ws] of this.activeStreams.entries()) {
      if (ws === client) {
        safeLog(this.logger, 'log', 'WebSocket disconnected', { callSid });
        const timer = this.silenceTimers.get(callSid);
        if (timer) { clearTimeout(timer); this.silenceTimers.delete(callSid); }
        this.activeStreams.delete(callSid);
        this.audioBuffers.delete(callSid);
        this.streamUsers.delete(callSid);
        this.rateLimit.delete(callSid);
        this.sessionService.delete(callSid).catch(() => {});
        this.activeGauge.dec();
        break;
      }
    }
  }

  @SubscribeMessage('message')
  async handleMessage(
    @MessageBody() data: string,
    @ConnectedSocket() client: WebSocket,
  ) {
    let callSid: string | null = null;
    try {
      const message: TwilioMediaMessage = JSON.parse(data);
      callSid = this.findCallSidByClient(client);
      if (callSid && !this.allowMessage(callSid)) {
        this.logger.warn(`Rate limit hit for call ${callSid}`);
        return;
      }

      switch (message.event) {
        case 'connected':
          safeLog(this.logger, 'log', 'Twilio Media Stream connected', {
            callSid,
          });
          break;

        case 'start':
          await this.handleStreamStart(message, client);
          break;

        case 'media':
          await this.handleMediaChunk(message, client);
          break;

        case 'stop':
          await this.handleStreamStop(message, client);
          break;

        case 'mark':
          // Mark events are used for timing
          this.logger.debug(`Mark: ${message.mark?.name}`);
          break;

        default: {
          // exhaustive check - this should never happen
          const exhaustiveCheck: never = message.event;
          this.logger.warn(`Unknown event type: ${String(exhaustiveCheck)}`);
        }
      }
    } catch (error) {
      safeLog(this.logger, 'error', 'Error handling WebSocket message', {
        callSid,
        error: error?.message,
      });
    }
  }

  private async handleStreamStart(
    message: TwilioMediaMessage,
    client: WebSocket,
  ) {
    const { callSid, streamSid, mediaFormat, customParameters } =
      message.start!;

    // Validate auth from Custom Parameters (NOT query string)
    const authResult = WsJwtGuard.validateTwilioStreamAuth(customParameters);
    if (!authResult.valid) {
      this.logger.warn(`Twilio stream auth failed: ${authResult.reason}`, {
        callSid,
      });
      client.close(4001, `Auth failed: ${authResult.reason}`);
      return;
    }

    safeLog(this.logger, 'log', 'Stream started (auth validated)', {
      streamSid,
      callSid,
      mediaFormat,
    });

    // Initialize audio buffer for this call
    this.audioBuffers.set(callSid, []);
    this.activeStreams.set(callSid, client);
    this.streamSids.set(callSid, streamSid);

    // Set up user context
    const user = { sub: `twilio:${callSid}`, roles: ['twilio'] };
    this.streamUsers.set(callSid, user);
    (client as any).user = user;
    (client as any).callSid = callSid;
    (client as any).twilio = true;

    // Persist session
    this.sessionService
      .create({
        userId: user.sub,
        callSid,
        metadata: {
          clinicianId: null,
          patientId: null,
          mode: 'voice_agent_va',
        },
      })
      .catch((e) =>
        this.logger.warn(`Session persist failed for ${callSid}: ${e}`),
      );

    this.activeGauge.inc();

    // Send acknowledgment
    client.send(
      JSON.stringify({
        event: 'connected',
        protocol: 'Call',
      }),
    );
  }

  private async handleMediaChunk(
    message: TwilioMediaMessage,
    client: WebSocket,
  ) {
    const { payload } = message.media!;
    const callSid = this.findCallSidByClient(client);
    const user = callSid ? this.streamUsers.get(callSid) : null;

    if (!callSid) {
      this.logger.warn('Received media chunk for unknown call');
      return;
    }

    // Decode base64 audio payload (mulaw format from Twilio)
    const audioChunk = Buffer.from(payload, 'base64');

    // Add to buffer
    const buffer = this.audioBuffers.get(callSid) || [];
    buffer.push(audioChunk);
    this.audioBuffers.set(callSid, buffer);

    const totalBytes = buffer.reduce((sum, c) => sum + c.length, 0);

    // Safety cap: drop buffer if we somehow exceed 24s of audio
    if (totalBytes >= VoiceGateway.MAX_BUFFER_BYTES) {
      this.audioBuffers.set(callSid, []);
      this.logger.warn(`Buffer overflow for ${callSid}: dropped ${totalBytes} bytes`);
      return;
    }

    // Only consider triggering once we have at least 1s of audio.
    // Each detected speech chunk resets (extends) the silence-gap timer so the
    // pipeline fires only after the caller has actually finished speaking.
    if (totalBytes >= VoiceGateway.MIN_BUFFER_BYTES && this.isSpeechChunk(audioChunk)) {
      this.scheduleSilenceTrigger(callSid);
    }
  }

  /**
   * Returns true if the G.711 PCMU chunk contains audible speech.
   * After removing the μ-law bit-flip (XOR 0x55), the exponent field
   * (bits 4-6) is 0 for near-silence and >0 for real audio energy.
   */
  private isSpeechChunk(chunk: Buffer): boolean {
    let active = 0;
    for (const b of chunk) {
      const m = b ^ 0x55;
      if (((m >> 4) & 0x7) > 0) active++;
    }
    return active / chunk.length > 0.20;
  }

  /**
   * (Re-)arm the silence-gap debounce timer for a call.
   * Fires SILENCE_GAP_MS after the last speech chunk is detected.
   * If the pipeline is still running when the timer fires, re-arms once
   * so the buffered audio is processed after the pipeline finishes.
   */
  private scheduleSilenceTrigger(callSid: string): void {
    const existing = this.silenceTimers.get(callSid);
    if (existing) clearTimeout(existing);

    const timer = setTimeout(() => {
      this.silenceTimers.delete(callSid);

      const buf = this.audioBuffers.get(callSid);
      if (!buf?.length) return;
      const totalBytes = buf.reduce((s, c) => s + c.length, 0);
      if (totalBytes < VoiceGateway.MIN_BUFFER_BYTES) return;

      if (this.processingCalls.has(callSid)) {
        // Pipeline still running — re-arm so buffered audio is caught when it finishes
        this.scheduleSilenceTrigger(callSid);
        return;
      }

      const combinedAudio = Buffer.concat(buf);
      this.audioBuffers.set(callSid, []);
      this.processAudioChunk(callSid, combinedAudio, false).catch((err) =>
        this.logger.error(`Silence-triggered ASR error for ${callSid}`, err),
      );
    }, VoiceGateway.SILENCE_GAP_MS);

    this.silenceTimers.set(callSid, timer);
  }

  private async handleStreamStop(
    message: TwilioMediaMessage,
    client: WebSocket,
  ) {
    const { callSid } = message.stop!;

    safeLog(this.logger, 'log', 'Stream stopped', { callSid });

    // Cancel any pending silence timer — we'll flush the buffer explicitly below
    const timer = this.silenceTimers.get(callSid);
    if (timer) { clearTimeout(timer); this.silenceTimers.delete(callSid); }

    // Process any remaining audio
    const buffer = this.audioBuffers.get(callSid);
    if (buffer && buffer.length > 0) {
      const combinedAudio = Buffer.concat(buffer);
      await this.processAudioChunk(callSid, combinedAudio, true);
    }

    // Cleanup
    this.activeStreams.delete(callSid);
    this.audioBuffers.delete(callSid);
    this.streamUsers.delete(callSid);
    this.rateLimit.delete(callSid);
    this.streamSids.delete(callSid);
  }

  private async processAudioChunk(
    callSid: string,
    audioData: Buffer,
    isFinal: boolean,
  ) {
    const user = this.streamUsers.get(callSid);
    this.processingCalls.add(callSid);
    try {
      // Convert mulaw to base64 for ASR service
      const base64Audio = audioData.toString('base64');

      // Directly run the full pipeline — silence-detection-based streaming
      // is unreliable on 8-bit mulaw (audioop RMS misinterpretation).
      const result = await this.conversationService.processVoiceInput({
        callSid,
        audio: base64Audio,
        format: 'mulaw',
        sampleRate: 8000,
        user,
      });

      if (result.transcript) {
        safeLog(this.logger, 'log', 'Transcript received', {
          callSid,
          length: result.transcript.length,
        });
        this.server.emit('final_transcript', {
          callSid,
          text: result.transcript,
        });
      }

      if (result.audioResponse) {
        await this.sendAudioToClient(callSid, result.audioResponse);
      }
    } catch (error) {
      safeLog(this.logger, 'error', 'Error processing audio chunk', {
        callSid,
        error: error?.message,
      });
      this.server.emit('transcript_error', {
        callSid,
        message: 'ASR unavailable',
      });
    } finally {
      this.processingCalls.delete(callSid);
    }
  }

  private async sendAudioToClient(callSid: string, audioData: string) {
    const client = this.activeStreams.get(callSid);

    if (!client || client.readyState !== WebSocket.OPEN) {
      this.logger.warn(`Cannot send audio to ${callSid}: client not connected`);
      return;
    }

    try {
      // Send audio back to Twilio as media message
      // Twilio requires the streamSid (MZ...), NOT the callSid (CA...)
      const streamSid = this.streamSids.get(callSid) || callSid;

      safeLog(this.logger, 'log', 'Sending audio to Twilio', {
        callSid,
        streamSid,
        byteLen: audioData.length,
        clientReadyState: client.readyState,
        hasMzSid: streamSid.startsWith('MZ'),
      });

      if (audioData.length < 500) {
        this.logger.warn(
          `Suspiciously small audio payload for ${callSid}: ${audioData.length} bytes — likely silence fallback`,
        );
      }

      const message = {
        event: 'media',
        streamSid,
        media: {
          payload: audioData,
        },
      };

      client.send(JSON.stringify(message));
      safeLog(this.logger, 'log', 'Audio sent to Twilio successfully', {
        callSid,
      });
    } catch (error) {
      this.logger.error('Error sending audio to client', error);
    }
  }

  private extractCallSidFromUrl(url: string): string | null {
    // Extract callSid from URL like /twilio/CAxxxx or /twilio/ws/CAxxxx
    const match = url.match(/\/twilio(?:\/ws)?\/([^?]+)/);
    return match ? match[1] : null;
  }

  private findCallSidByClient(client: WebSocket): string | null {
    for (const [callSid, ws] of this.activeStreams.entries()) {
      if (ws === client) {
        return callSid;
      }
    }
    return null;
  }

  private allowMessage(callSid: string): boolean {
    const now = Math.floor(Date.now() / 1000);
    const entry = this.rateLimit.get(callSid) || { count: 0, ts: now };
    if (entry.ts !== now) {
      entry.count = 0;
      entry.ts = now;
    }
    entry.count += 1;
    this.rateLimit.set(callSid, entry);
    return entry.count <= 50; // max 50 messages per second per call
  }
}
