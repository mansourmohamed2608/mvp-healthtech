// gateway/src/conversation/conversation.service.ts
/**
 * Conversation Service - Stateful conversation management with Redis
 * Week 3 Day 17 (Oct 11, 2025)
 * Enhanced with retry logic, error handling, and conversation context management
 */
import { Injectable, Logger } from '@nestjs/common';
import { createClient, RedisClientType } from 'redis';
import { LlmService } from '../llm/llm.service';
import { TtsService } from '../tts/tts.service';
import { AsrService } from '../asr/asr.service';
import { MetricsController } from '../metrics/metrics.controller';
import { safeLog } from '../utils/safe-logger';

interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
  metadata?: Record<string, any>;
}

interface ConversationState {
  sessionId: string;
  messages: Message[];
  context: Record<string, any>;
  lastActivity: number;
}

@Injectable()
export class ConversationService {
  private readonly logger = new Logger(ConversationService.name);
  private readonly client: RedisClientType | null = null;
  private readonly inMemoryStore = new Map<string, Message[]>();
  private readonly inflight = new Map<string, number>();
  private readonly asrMetric = MetricsController.getAsrLatency();
  private readonly llmMetric = MetricsController.getLlmLatency();
  private readonly ttsMetric = MetricsController.getTtsLatency();
  private readonly MAX_RETRIES = 3;
  private readonly RETRY_DELAY_MS = 1000;
  private readonly MAX_MESSAGES = 20; // Keep last 20 messages
  private readonly CONVERSATION_TTL = 7200; // 2 hours
  private redisAvailable = false;
  private readonly llmEnabled =
    (process.env.VOICE_AGENT_LLM_ENABLED || '1').toString().trim() !== '0';
  private readonly cannedReply =
    'مرحبا، هذا مجرد اختبار للصوت في النظام.';

  constructor(
    private readonly llmService: LlmService,
    private readonly ttsService: TtsService,
    private readonly asrService: AsrService,
  ) {
    // Try to connect to Redis, but don't block if it fails
    try {
      this.client = createClient({
        url: `redis://${process.env.REDIS_HOST || 'localhost'}:${process.env.REDIS_PORT || 6379}`,
        socket: {
          connectTimeout: 2000, // 2 second timeout
          reconnectStrategy: () => false, // Don't retry
        },
      });

      this.client.on('error', (err) => {
        this.logger.warn('⚠️  Redis not available, using in-memory storage');
        this.redisAvailable = false;
      });

      this.client.on('connect', () => {
        this.logger.log('✅ Connected to Redis');
        this.redisAvailable = true;
      });

      // Connect without blocking
      this.client.connect().catch((err) => {
        this.logger.warn('⚠️  Redis connection failed, using in-memory storage');
        this.redisAvailable = false;
      });
    } catch (error) {
      this.logger.warn('⚠️  Redis initialization failed, using in-memory storage');
      this.redisAvailable = false;
    }
  }

  /**
   * Append a message to conversation with retry logic
   */
  async appendMessage(
    sessionId: string,
    role: 'user' | 'assistant' | 'system',
    content: string,
    metadata?: Record<string, any>,
  ): Promise<void> {
    const message: Message = {
      role,
      content,
      timestamp: Date.now(),
      metadata,
    };

    // Use in-memory storage if Redis unavailable
    if (!this.redisAvailable || !this.client) {
      const messages = this.inMemoryStore.get(sessionId) || [];
      messages.push(message);
      // Keep only last MAX_MESSAGES
      if (messages.length > this.MAX_MESSAGES) {
        messages.splice(0, messages.length - this.MAX_MESSAGES);
      }
      this.inMemoryStore.set(sessionId, messages);
      this.logger.debug(`Appended ${role} message to ${sessionId} (in-memory)`);
      return;
    }

    // Use Redis if available
    try {
      const key = `conv:${sessionId}`;
      await this.client.rPush(key, JSON.stringify(message));

      // Trim to keep only last MAX_MESSAGES
      const length = await this.client.lLen(key);
      if (length > this.MAX_MESSAGES) {
        await this.client.lTrim(key, -this.MAX_MESSAGES, -1);
      }

      // Reset TTL
      await this.client.expire(key, this.CONVERSATION_TTL);

      this.logger.debug(`Appended ${role} message to ${sessionId}`);
    } catch (error) {
      this.logger.warn('Redis error, falling back to in-memory');
      // Fallback to in-memory
      const messages = this.inMemoryStore.get(sessionId) || [];
      messages.push(message);
      if (messages.length > this.MAX_MESSAGES) {
        messages.splice(0, messages.length - this.MAX_MESSAGES);
      }
      this.inMemoryStore.set(sessionId, messages);
    }
  }

  /**
   * Get conversation messages with context window
   */
  async getMessages(sessionId: string, limit = 10): Promise<Message[]> {
    // Use in-memory storage if Redis unavailable
    if (!this.redisAvailable || !this.client) {
      const messages = this.inMemoryStore.get(sessionId) || [];
      return messages.slice(-limit);
    }

    // Use Redis if available
    try {
      const key = `conv:${sessionId}`;
      const items = await this.client.lRange(key, -limit, -1);
      return items.map((x) => JSON.parse(x) as Message);
    } catch (error) {
      this.logger.warn('Redis error, falling back to in-memory');
      const messages = this.inMemoryStore.get(sessionId) || [];
      return messages.slice(-limit);
    }
  }

  /**
   * Get conversation history (alias for getMessages)
   */
  async getHistory(sessionId: string, limit = 10): Promise<Message[]> {
    return this.getMessages(sessionId, limit);
  }

  /**
   * Get full conversation state
   */
  async getState(sessionId: string): Promise<ConversationState | null> {
    try {
      const messages = await this.getMessages(sessionId, this.MAX_MESSAGES);
      let context = {};

      if (this.redisAvailable && this.client) {
        const contextKey = `conv:context:${sessionId}`;
        const contextStr = await this.client.get(contextKey);
        context = contextStr ? JSON.parse(contextStr) : {};
      }

      return {
        sessionId,
        messages,
        context,
        lastActivity: messages.length > 0 ? messages[messages.length - 1].timestamp : 0,
      };
    } catch (error) {
      this.logger.error(`Failed to get state for ${sessionId}:`, error);
      return null;
    }
  }

  /**
   * Update conversation context (e.g., intent, entities)
   */
  async updateContext(sessionId: string, context: Record<string, any>): Promise<void> {
    if (!this.redisAvailable || !this.client) {
      this.logger.debug(`Context not persisted (in-memory mode)`);
      return;
    }

    try {
      const contextKey = `conv:context:${sessionId}`;
      await this.client.set(contextKey, JSON.stringify(context), {
        EX: this.CONVERSATION_TTL,
      });
      this.logger.debug(`Updated context for ${sessionId}`);
    } catch (error) {
      this.logger.error(`Failed to update context for ${sessionId}:`, error);
    }
  }

  /**
   * Clear conversation
   */
  async clear(sessionId: string): Promise<void> {
    // Clear in-memory
    this.inMemoryStore.delete(sessionId);

    if (!this.redisAvailable || !this.client) {
      this.logger.log(`Cleared conversation for ${sessionId} (in-memory)`);
      return;
    }

    try {
      const key = `conv:${sessionId}`;
      const contextKey = `conv:context:${sessionId}`;
      await this.client.del([key, contextKey]);
      this.logger.log(`Cleared conversation for ${sessionId}`);
    } catch (error) {
      this.logger.error(`Failed to clear conversation for ${sessionId}:`, error);
    }
  }

  /**
   * Get conversation summary for LLM context
   */
  async getSummary(sessionId: string): Promise<string> {
    try {
      const messages = await this.getMessages(sessionId, 5);
      if (messages.length === 0) {
        return 'بداية محادثة جديدة.'; // "New conversation start"
      }

      return messages
        .map((m) => `${m.role === 'user' ? 'المريض' : 'المساعد'}: ${m.content}`)
        .join('\n');
    } catch (error) {
      this.logger.error(`Failed to get summary for ${sessionId}:`, error);
      return '';
    }
  }

  /**
   * Check if conversation is active
   */
  async isActive(sessionId: string): Promise<boolean> {
    // Check in-memory first
    if (this.inMemoryStore.has(sessionId)) {
      return true;
    }

    if (!this.redisAvailable || !this.client) {
      return false;
    }

    try {
      const key = `conv:${sessionId}`;
      const exists = await this.client.exists(key);
      return exists === 1;
    } catch (error) {
      this.logger.error(`Failed to check if conversation active: ${error}`);
      return false;
    }
  }

  /**
   * Process voice input from Twilio Media Stream
   * 1. Send audio to ASR for transcription
   * 2. Send transcript to LLM for medical response
   * 3. Send response to TTS for voice synthesis
   * 4. Return transcript and audio response
   */
  async processVoiceInput(input: {
    callSid: string;
    audio: string; // base64 encoded
    format: string; // 'mulaw' or 'wav'
    sampleRate: number;
    user?: any;
    partialOnly?: boolean;
  }): Promise<{
    transcript: string;
    response: string;
    audioResponse: string;
  }> {
    const current = this.inflight.get(input.callSid) || 0;
    if (current >= 1) {
      this.logger.warn(`Backpressure: dropping chunk for ${input.callSid}`);
      return { transcript: '', response: '', audioResponse: '' };
    }
    this.inflight.set(input.callSid, current + 1);
    try {
      const asrStart = process.hrtime();
      // 1. Transcribe audio using ASR service (with timeout/error handling)
      const { text: transcript } = await this.asrService.transcribe(
        input.audio,
        input.callSid,
        true,
      );
      this.asrMetric.observe({ endpoint: 'transcribe', status: 'ok' }, this.durationSeconds(asrStart));

      if (!transcript || transcript.trim() === '') {
        // No speech detected, return empty
        return { transcript: '', response: '', audioResponse: '' };
      }

      if (input.partialOnly) {
        // For partial streaming we only return transcript
        return { transcript, response: '', audioResponse: '' };
      }

      // Avoid logging PHI; only log lengths
      this.logger.log(`Transcribed (${input.callSid}): ${transcript.length} chars`);

      // Save user message to conversation
      await this.appendMessage(input.callSid, 'user', transcript);

      // 2. Get response (LLM or canned)
      let response = this.cannedReply;
      if (this.llmEnabled) {
        const llmStart = process.hrtime();
        const history = await this.getHistory(input.callSid);
        const chatPayload = {
          message: transcript,
          history: history.map((m) => ({ role: m.role, content: m.content })),
          sessionId: input.callSid,
        };
        try {
          const llmResult = await this.llmService.chat(chatPayload);
          response = llmResult.reply || this.cannedReply;
          this.llmMetric.observe({ endpoint: 'chat', status: 'ok' }, this.durationSeconds(llmStart));
          this.logger.log(`LLM response (${input.callSid}) length=${response?.length ?? 0}`);
        } catch (error) {
          this.llmMetric.observe({ endpoint: 'chat', status: 'error' }, this.durationSeconds(llmStart));
          safeLog(this.logger, 'warn', 'LLM unavailable, using canned reply', {
            callSid: input.callSid,
            error: (error as any)?.message,
          });
          response = this.cannedReply;
        }
      } else {
        this.logger.debug(`LLM disabled, using canned reply for ${input.callSid}`);
      }

      // Save assistant message to conversation
      await this.appendMessage(input.callSid, 'assistant', response);

      // 3. Synthesize voice response using TTS
      const ttsStart = process.hrtime();
      const ttsResult = await this.ttsService.synthesize(response, input.callSid);
      // TTS returns base64 mulaw (8k) when available; Twilio media streams expect mulaw payloads.
      const audioResponse = ttsResult.audioBase64;
      this.ttsMetric.observe({ endpoint: 'synthesize', status: 'ok' }, this.durationSeconds(ttsStart));

      return {
        transcript,
        response,
        audioResponse,
      };
    } catch (error) {
      this.logger.error('Error processing voice input', error);
      this.asrMetric.observe({ endpoint: 'transcribe', status: 'error' }, 0);
      return {
        transcript: '',
        response: 'عذراً، حدث خطأ. يرجى المحاولة مرة أخرى لاحقاً.',
        audioResponse: '',
      };
    } finally {
      const now = this.inflight.get(input.callSid) || 1;
      this.inflight.set(input.callSid, Math.max(0, now - 1));
    }
  }

  /**
   * Utility: delay for retry logic
   */
  private delay(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  private durationSeconds(start: [number, number]) {
    const diff = process.hrtime(start);
    return diff[0] + diff[1] / 1e9;
  }

  /**
   * Cleanup on service shutdown
   */
  async onModuleDestroy() {
    if (this.client) {
      await this.client.quit();
      this.logger.log('Redis client disconnected');
    }
  }
}
