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

interface TwilioMediaMessage {
  event: 'connected' | 'start' | 'media' | 'stop' | 'mark';
  streamSid?: string;
  start?: {
    streamSid: string;
    accountSid: string;
    callSid: string;
    tracks: string[];
    mediaFormat: {
      encoding: string;
      sampleRate: number;
      channels: number;
    };
  };
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

// WS auth: JWT + HMAC(sig, ts, callSid) derived from TWILIO_AUTH_TOKEN/WS_SHARED_SECRET.
@UseGuards(WsJwtGuard)
@WebSocketGateway({
  path: '/twilio/ws',
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
  private readonly activeGauge = MetricsController.getActiveConversations();

  constructor(
    private readonly conversationService: ConversationService,
    private readonly sessionService: SessionService,
    private readonly asrService: AsrService,
  ) {}

  handleConnection(client: WebSocket, request: any) {
    const callSid = this.extractCallSidFromUrl(request.url);
    safeLog(this.logger, 'log', 'WebSocket connected', { callSid: callSid || 'unknown' });
    // Attach user claims from guard
    const user = (client as any).user || {};

    // Require authenticated WS
    if (!user || !user.sub) {
      this.logger.warn('Unauthorized WS connection attempt');
      client.close();
      return;
    }

    if (callSid) {
      this.activeStreams.set(callSid, client);
      this.audioBuffers.set(callSid, []);
      this.streamUsers.set(callSid, user);
      // persist session
      this.sessionService.create({
        userId: user.sub,
        callSid,
        metadata: {
          clinicianId: user.sub,
          patientId: user.patientId || null,
        },
      }).catch((e) => this.logger.warn(`Session persist failed for ${callSid}: ${e}`));
      this.activeGauge.inc();
    }
  }

  handleDisconnect(client: WebSocket) {
    // Find and remove the disconnected client
    for (const [callSid, ws] of this.activeStreams.entries()) {
      if (ws === client) {
        safeLog(this.logger, 'log', 'WebSocket disconnected', { callSid });
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
          safeLog(this.logger, 'log', 'Twilio Media Stream connected', { callSid });
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

        default:
          this.logger.warn(`Unknown event type: ${message.event}`);
      }
    } catch (error) {
      safeLog(this.logger, 'error', 'Error handling WebSocket message', { callSid, error: (error as any)?.message });
    }
  }

  private async handleStreamStart(
    message: TwilioMediaMessage,
    client: WebSocket,
  ) {
    const { callSid, streamSid, mediaFormat } = message.start!;
    
    safeLog(this.logger, 'log', 'Stream started', { streamSid, callSid, mediaFormat });

    // Initialize audio buffer for this call
    this.audioBuffers.set(callSid, []);
    this.activeStreams.set(callSid, client);

    // Send acknowledgment
    client.send(JSON.stringify({
      event: 'connected',
      protocol: 'Call',
    }));
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

    // Process audio every ~300ms (approximately 2400-4800 bytes for mulaw at 8kHz)
    const totalBytes = buffer.reduce((sum, chunk) => sum + chunk.length, 0);
    
    if (totalBytes >= 2400) { // ~300ms of audio
      // Combine all chunks
      const combinedAudio = Buffer.concat(buffer);
      
      // Clear buffer
      this.audioBuffers.set(callSid, []);

      // Send to conversation service for transcription and processing
      try {
        await this.processAudioChunk(callSid, combinedAudio, false);
      } catch (error) {
        this.logger.error(`Error processing audio chunk for ${callSid}`, error);
      }
    }
  }

  private async handleStreamStop(
    message: TwilioMediaMessage,
    client: WebSocket,
  ) {
    const { callSid } = message.stop!;
    
    safeLog(this.logger, 'log', 'Stream stopped', { callSid });

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
  }

  private async processAudioChunk(callSid: string, audioData: Buffer, isFinal: boolean) {
    const user = this.streamUsers.get(callSid);
    try {
      // Convert mulaw to base64 for ASR service
      const base64Audio = audioData.toString('base64');

      const streamResp = await this.asrService.stream(base64Audio, callSid, isFinal);

      if (streamResp.partial) {
        this.server.emit('partial_transcript', { callSid, text: streamResp.partial });
      }

      // If final or stream stop, run full pipeline
      if (isFinal || streamResp.isFinal) {
        const result = await this.conversationService.processVoiceInput({
          callSid,
          audio: base64Audio,
          format: 'mulaw',
          sampleRate: 8000,
          user,
        });

        if (result.transcript) {
          safeLog(this.logger, 'log', 'Transcript received', { callSid, length: result.transcript.length });
          this.server.emit('final_transcript', { callSid, text: result.transcript });
        }

        if (result.audioResponse) {
          await this.sendAudioToClient(callSid, result.audioResponse);
        }
      }
    } catch (error) {
      safeLog(this.logger, 'error', 'Error processing audio chunk', { callSid, error: (error as any)?.message });
      this.server.emit('transcript_error', { callSid, message: 'ASR unavailable' });
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
      // Twilio expects base64 mulaw audio
      const message = {
        event: 'media',
        streamSid: callSid,
        media: {
          payload: audioData,
        },
      };

      client.send(JSON.stringify(message));
    } catch (error) {
      this.logger.error('Error sending audio to client', error);
    }
  }

  private extractCallSidFromUrl(url: string): string | null {
    // Extract callSid from URL like /twilio/ws/CAxxxx
    const match = url.match(/\/twilio\/ws\/([^?]+)/);
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
