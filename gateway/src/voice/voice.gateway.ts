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
import { Logger } from '@nestjs/common';
import { Server, WebSocket } from 'ws';
import { ConversationService } from '../conversation/conversation.service';
import { SessionService } from '../session/session.service';

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

  constructor(
    private readonly conversationService: ConversationService,
    private readonly sessionService: SessionService,
  ) {}

  handleConnection(client: WebSocket, request: any) {
    const callSid = this.extractCallSidFromUrl(request.url);
    this.logger.log(`WebSocket connected: ${callSid || 'unknown'}`);
    
    if (callSid) {
      this.activeStreams.set(callSid, client);
      this.audioBuffers.set(callSid, []);
    }
  }

  handleDisconnect(client: WebSocket) {
    // Find and remove the disconnected client
    for (const [callSid, ws] of this.activeStreams.entries()) {
      if (ws === client) {
        this.logger.log(`WebSocket disconnected: ${callSid}`);
        this.activeStreams.delete(callSid);
        this.audioBuffers.delete(callSid);
        break;
      }
    }
  }

  @SubscribeMessage('message')
  async handleMessage(
    @MessageBody() data: string,
    @ConnectedSocket() client: WebSocket,
  ) {
    try {
      const message: TwilioMediaMessage = JSON.parse(data);

      switch (message.event) {
        case 'connected':
          this.logger.log('Twilio Media Stream connected');
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
      this.logger.error('Error handling WebSocket message', error);
    }
  }

  private async handleStreamStart(
    message: TwilioMediaMessage,
    client: WebSocket,
  ) {
    const { callSid, streamSid, mediaFormat } = message.start!;
    
    this.logger.log(`Stream started: ${streamSid} for call: ${callSid}`);
    this.logger.log(`Media format: ${JSON.stringify(mediaFormat)}`);

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

    // Process audio every ~300ms (approximately 4800 bytes for mulaw at 8kHz)
    const totalBytes = buffer.reduce((sum, chunk) => sum + chunk.length, 0);
    
    if (totalBytes >= 2400) { // ~300ms of audio
      // Combine all chunks
      const combinedAudio = Buffer.concat(buffer);
      
      // Clear buffer
      this.audioBuffers.set(callSid, []);

      // Send to conversation service for transcription and processing
      try {
        await this.processAudioChunk(callSid, combinedAudio);
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
    
    this.logger.log(`Stream stopped for call: ${callSid}`);

    // Process any remaining audio
    const buffer = this.audioBuffers.get(callSid);
    if (buffer && buffer.length > 0) {
      const combinedAudio = Buffer.concat(buffer);
      await this.processAudioChunk(callSid, combinedAudio);
    }

    // Cleanup
    this.activeStreams.delete(callSid);
    this.audioBuffers.delete(callSid);
  }

  private async processAudioChunk(callSid: string, audioData: Buffer) {
    try {
      // Convert mulaw to base64 for ASR service
      const base64Audio = audioData.toString('base64');

      // Send to conversation service which will:
      // 1. Forward to ASR for transcription
      // 2. Send transcript to LLM for response
      // 3. Send response to TTS for voice
      // 4. Return voice audio to play back
      const result = await this.conversationService.processVoiceInput({
        callSid,
        audio: base64Audio,
        format: 'mulaw',
        sampleRate: 8000,
      });

      if (result.transcript) {
        this.logger.log(`Transcript (${callSid}): ${result.transcript}`);
      }

      // If we have a voice response, send it back to Twilio
      if (result.audioResponse) {
        await this.sendAudioToClient(callSid, result.audioResponse);
      }
    } catch (error) {
      this.logger.error('Error processing audio chunk', error);
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
}
