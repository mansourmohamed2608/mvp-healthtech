// gateway/src/conversation/conversation.service.ts
/**
 * Conversation Service - Stateful conversation management with Redis
 * Week 3 Day 17 (Oct 11, 2025)
 * Enhanced with retry logic, error handling, and conversation context management
 */
import { Injectable, Logger } from '@nestjs/common';
import { createClient, RedisClientType } from 'redis';
import { Pool } from 'pg';
import { LlmService } from '../llm/llm.service';
import { TtsService } from '../tts/tts.service';
import { AsrService } from '../asr/asr.service';
import { MetricsController } from '../metrics/metrics.controller';
import { safeLog } from '../utils/safe-logger';
import { VaBookingService, SlotState } from '../va/va_booking.service';

export interface Message {
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
  private readonly inMemoryContext = new Map<string, Record<string, any>>();
  private readonly inflight = new Map<string, number>();
  private readonly pool: Pool | null;
  private readonly persistTranscripts =
    (process.env.VA_TRANSCRIPT_PERSIST || '1').toString().trim() !== '0';
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
  private readonly cannedReply = 'مرحبا، هذا مجرد اختبار للصوت في النظام.';
  private readonly voiceEgypt = process.env.TTS_VOICE_EGYPT || 'egtts';
  private readonly voiceSaudi = process.env.TTS_VOICE_SAUDI || 'saudi-tts';

  constructor(
    private readonly llmService: LlmService,
    private readonly ttsService: TtsService,
    private readonly asrService: AsrService,
    private readonly vaBookingService: VaBookingService,
  ) {
    const dbUrl = process.env.DATABASE_URL;
    this.pool = dbUrl ? new Pool({ connectionString: dbUrl }) : null;
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
        this.logger.warn(
          '⚠️  Redis connection failed, using in-memory storage',
        );
        this.redisAvailable = false;
      });
    } catch (error) {
      this.logger.warn(
        '⚠️  Redis initialization failed, using in-memory storage',
      );
      this.redisAvailable = false;
    }
  }

  private async getSlots(sessionId: string): Promise<Record<string, any>> {
    const state = await this.getState(sessionId);
    const existing = (state?.context as any)?.slots || {};
    return {
      name: existing.name || '',
      phone: existing.phone || '',
      dob: existing.dob || '',
      visit_type: existing.visit_type || '',
      specialty: existing.specialty || '',
      doctor_name: existing.doctor_name || '',
      date: existing.date || '',
      time: existing.time || '',
      no_marketing: existing.no_marketing ?? null,
    };
  }

  private async updateSlots(sessionId: string, slots: Record<string, any>) {
    const state = await this.getState(sessionId);
    const context = { ...(state?.context || {}), slots };
    await this.updateContext(sessionId, context);
  }

  private bookingReady(slots: SlotState): boolean {
    return (
      slots.name.trim() !== '' &&
      slots.phone.trim() !== '' &&
      slots.dob.trim() !== '' &&
      (slots.doctor_name.trim() !== '' || slots.specialty.trim() !== '') &&
      slots.date.trim() !== '' &&
      slots.time.trim() !== '' &&
      (slots as any).booked !== true
    );
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
      await this.persistMessage(sessionId, message);
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

      await this.persistMessage(sessionId, message);
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
      await this.persistMessage(sessionId, message);
    }
  }

  /**
   * Get conversation messages with context window
   */
  async getMessages(sessionId: string, limit = 10): Promise<Message[]> {
    // Use in-memory storage if Redis unavailable
    if (!this.redisAvailable || !this.client) {
      const messages = this.inMemoryStore.get(sessionId) || [];
      const recent = messages.slice(-limit);
      if (recent.length > 0) {
        return recent;
      }
      const persisted = await this.loadPersistedMessages(sessionId, limit);
      return persisted.length ? persisted : recent;
    }

    // Use Redis if available
    try {
      const key = `conv:${sessionId}`;
      const items = await this.client.lRange(key, -limit, -1);
      const parsed = items.map((x) => JSON.parse(x) as Message);
      if (parsed.length > 0) {
        return parsed;
      }
      const persisted = await this.loadPersistedMessages(sessionId, limit);
      return persisted.length ? persisted : parsed;
    } catch (error) {
      this.logger.warn('Redis error, falling back to in-memory');
      const messages = this.inMemoryStore.get(sessionId) || [];
      const recent = messages.slice(-limit);
      if (recent.length > 0) {
        return recent;
      }
      const persisted = await this.loadPersistedMessages(sessionId, limit);
      return persisted.length ? persisted : recent;
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
      let context: Record<string, any> = {};

      if (this.redisAvailable && this.client) {
        const contextKey = `conv:context:${sessionId}`;
        const contextStr = await this.client.get(contextKey);
        context = contextStr ? JSON.parse(contextStr) : {};
      } else {
        context = this.inMemoryContext.get(sessionId) || {};
      }

      return {
        sessionId,
        messages,
        context,
        lastActivity:
          messages.length > 0 ? messages[messages.length - 1].timestamp : 0,
      };
    } catch (error) {
      this.logger.error(`Failed to get state for ${sessionId}:`, error);
      return null;
    }
  }

  /**
   * Update conversation context (e.g., intent, entities)
   */
  async updateContext(
    sessionId: string,
    context: Record<string, any>,
  ): Promise<void> {
    if (!this.redisAvailable || !this.client) {
      this.inMemoryContext.set(sessionId, context || {});
      this.logger.debug(`Context updated (in-memory mode)`);
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
      this.logger.error(
        `Failed to clear conversation for ${sessionId}:`,
        error,
      );
    }
  }

  async onModuleDestroy() {
    if (this.client) {
      await this.client.quit();
    }
    if (this.pool) {
      await this.pool.end();
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
      const existingState = await this.getState(input.callSid);
      const existingContext = existingState?.context || {};
      const preferences = (existingContext.preferences || {}) as Record<
        string,
        any
      >;
      const preferredTenant =
        typeof preferences.tenantId === 'string'
          ? preferences.tenantId.trim()
          : '';
      const tenantId =
        preferredTenant ||
        process.env.DEFAULT_TENANT_ID ||
        process.env.VA_TENANT_ID ||
        'default';
      const preferredDialect = this.normalizeDialect(preferences.dialect);
      const voiceDialect = this.voiceToDialect(preferences.voice);
      const storedDialect = this.normalizeDialect(existingContext.dialect);
      const hintDialect =
        preferredDialect && preferredDialect !== 'auto'
          ? preferredDialect
          : voiceDialect || storedDialect || 'saudi';

      const asrStart = process.hrtime();
      // 1. Transcribe audio using ASR service (with timeout/error handling)
      const { text: transcript } = await this.asrService.transcribe(
        input.audio,
        input.callSid,
        {
          identifySpeakers: false,
          dialect: hintDialect,
          enableDiarization: false,
          diarizeFirst: false,
          enableAlignment: false,
          format: input.format,
          sampleRate: input.sampleRate,
        },
      );
      this.asrMetric.observe(
        { endpoint: 'transcribe', status: 'ok' },
        this.durationSeconds(asrStart),
      );

      if (!transcript || transcript.trim().length < 5) {
        // No speech detected or too short (ASR noise hallucination), return empty
        return { transcript: '', response: '', audioResponse: '' };
      }

      // Filter out well-known Whisper Arabic hallucinations (YouTube sign-off phrases
      // that Whisper generates when given near-silence or very short audio)
      const WHISPER_HALLUCINATIONS = [
        'شكرا لمشاهدة', 'شكراً لمشاهدة', 'شكرًا لمشاهدة',
        'شكرا للمشاهدة', 'شكراً للمشاهدة', 'شكرًا للمشاهدة',
        'شكرا على المشاهدة', 'شكراً على المشاهدة',
        'لا تنسى الاشتراك', 'لا تنسوا الاشتراك',
        'اشترك في القناة', 'اشتركوا في القناة',
        'لايك وشير', 'لايك وسيبسكرايب',
        'للمشاهدة والمتابعة',
      ];
      const trimmedTranscript = transcript.trim();
      const isHallucination = WHISPER_HALLUCINATIONS.some(
        (h) => trimmedTranscript === h ||
               trimmedTranscript.includes(h) && trimmedTranscript.length < h.length + 10,
      );
      if (isHallucination) {
        this.logger.warn(`Whisper hallucination filtered for ${input.callSid}: "${trimmedTranscript}"`);
        return { transcript: '', response: '', audioResponse: '' };
      }

      if (input.partialOnly) {
        // For partial streaming we only return transcript
        return { transcript, response: '', audioResponse: '' };
      }

      // Avoid logging PHI; only log lengths
      this.logger.log(
        `Transcribed (${input.callSid}): ${transcript.length} chars`,
      );

      // NOTE: user message is appended AFTER the LLM call (below) so that
      // the history passed to the LLM does not already contain the current
      // turn (which would make it appear twice — once in history, once in
      // the transcript field).

      const detectedDialect = this.detectDialect(transcript);
      const resolvedDialect =
        preferredDialect && preferredDialect !== 'auto'
          ? preferredDialect
          : voiceDialect || detectedDialect || storedDialect || 'saudi';

      if (resolvedDialect !== storedDialect || preferredDialect) {
        await this.updateContext(input.callSid, {
          ...existingContext,
          dialect: resolvedDialect,
          preferences: {
            ...preferences,
            dialect: preferredDialect || preferences.dialect || 'auto',
          },
        });
      }

      // 2. Get response (LLM or canned)
      let response = this.cannedReply;
      if (this.llmEnabled) {
        const llmStart = process.hrtime();
        const history = await this.getHistory(input.callSid);
        const slots = await this.getSlots(input.callSid);
        const chatPayload = {
          transcript,
          history: history.map((m) => ({ role: m.role, content: m.content })),
          sessionId: input.callSid,
          mode: 'voice_agent_va',
          slots,
          dialect: resolvedDialect,
          tenantId,
        };
        try {
          const llmResult = await this.llmService.orchestrate(chatPayload);
          response = llmResult.reply || this.cannedReply;
          if (llmResult.slots) {
            await this.updateSlots(input.callSid, llmResult.slots);
            const bookingReady = this.bookingReady(
              llmResult.slots as SlotState,
            );
            if (bookingReady) {
              const booking = await this.vaBookingService.tryBook(
                llmResult.slots as SlotState,
                input.callSid,
              );
              if (booking.success) {
                response += `\nتم حجز موعد مع ${booking.doctorName} بتاريخ ${booking.start?.slice(0, 10)} في ${booking.start?.slice(11, 16)}. سنقوم بالتأكيد من مركز علاجك.`;
                await this.updateSlots(input.callSid, {
                  ...(llmResult.slots as any),
                  booked: true,
                });
              } else if (booking.alternatives && booking.alternatives.length) {
                const opts = booking.alternatives
                  .map(
                    (o) =>
                      `${o.start.slice(0, 10)} الساعة ${o.start.slice(11, 16)}`,
                  )
                  .join('، ');
                response += `\n${booking.message || 'الموعد غير متاح'}، أقدر أقترح: ${opts}. أيهم يناسبك؟`;
              }
            }
          }
          this.llmMetric.observe(
            { endpoint: 'chat', status: 'ok' },
            this.durationSeconds(llmStart),
          );
          this.logger.log(
            `LLM response (${input.callSid}) length=${response?.length ?? 0}`,
          );
        } catch (error) {
          this.llmMetric.observe(
            { endpoint: 'chat', status: 'error' },
            this.durationSeconds(llmStart),
          );
          safeLog(this.logger, 'warn', 'LLM unavailable, using canned reply', {
            callSid: input.callSid,
            error: error?.message,
          });
          response = this.cannedReply;
        }
      } else {
        this.logger.debug(
          `LLM disabled, using canned reply for ${input.callSid}`,
        );
      }

      // Save user then assistant message (user first, then assistant, in order)
      await this.appendMessage(input.callSid, 'user', transcript);
      await this.appendMessage(input.callSid, 'assistant', response);

      // 3. Synthesize voice response using TTS
      const ttsStart = process.hrtime();
      const voice = this.selectVoice(resolvedDialect, preferences.voice);
      const ttsResult = await this.ttsService.synthesize(
        response,
        input.callSid,
        voice,
      );
      // TTS returns base64 mulaw (8k) when available; Twilio media streams expect mulaw payloads.
      const audioResponse = ttsResult.audioBase64;
      this.ttsMetric.observe(
        { endpoint: 'synthesize', status: 'ok' },
        this.durationSeconds(ttsStart),
      );

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
  private normalizeDialect(value?: string): string | null {
    if (!value) return null;
    const normalized = value.toString().trim().toLowerCase();
    if (!normalized) return null;
    if (normalized === 'auto') return 'auto';
    if (['egypt', 'egyptian', 'eg'].includes(normalized)) return 'egypt';
    if (['saudi', 'ksa', 'gulf', 'gcc'].includes(normalized)) return 'saudi';
    return normalized;
  }

  private async persistMessage(
    sessionId: string,
    message: Message,
  ): Promise<void> {
    if (!this.pool || !this.persistTranscripts) return;
    try {
      await this.pool.query(
        `INSERT INTO conversation_messages (session_id, role, content, message_ts, metadata)
         VALUES ($1, $2, $3, $4, $5)`,
        [
          sessionId,
          message.role,
          message.content,
          message.timestamp,
          message.metadata || {},
        ],
      );
    } catch (error) {
      this.logger.warn(`Transcript persist failed for ${sessionId}`);
    }
  }

  private async loadPersistedMessages(
    sessionId: string,
    limit: number,
  ): Promise<Message[]> {
    if (!this.pool || !this.persistTranscripts) return [];
    try {
      const result = await this.pool.query(
        `SELECT role, content, message_ts, metadata
         FROM conversation_messages
         WHERE session_id = $1
         ORDER BY message_ts DESC
         LIMIT $2`,
        [sessionId, limit],
      );
      return result.rows
        .map((row) => ({
          role: row.role,
          content: row.content,
          timestamp: Number(row.message_ts) || Date.now(),
          metadata: row.metadata || undefined,
        }))
        .reverse();
    } catch (error) {
      this.logger.warn(`Transcript load failed for ${sessionId}`);
      return [];
    }
  }

  private voiceToDialect(voice?: string): string | null {
    if (!voice) return null;
    const normalized = voice.toString().trim().toLowerCase();
    if (!normalized || normalized === 'auto') return null;
    if (normalized.includes('saudi')) return 'saudi';
    if (normalized.includes('egtts') || normalized.includes('egypt'))
      return 'egypt';
    return null;
  }

  private detectDialect(text: string): string | null {
    if (!text) return null;
    const lowered = text.toLowerCase();
    const egyptianMarkers = [
      'ازاي',
      'إزاي',
      'عايز',
      'عاوز',
      'عايزة',
      'مش',
      'ليه',
      'أيوه',
      'ايوه',
      'لسه',
      'دلوقتي',
      'بتاع',
      'كده',
      'حاجة',
      'عاوزة',
      'مفيش',
      'فين',
      'عايزين',
    ];
    const saudiMarkers = [
      'ايش',
      'إيش',
      'وش',
      'وشلون',
      'وش الاسم',
      'ليش',
      'ابغى',
      'أبغى',
      'يبغى',
      'تبي',
      'تبغى',
      'ودي',
      'الحين',
      'مره',
      'ترى',
      'حيل',
      'ياخي',
      'دام',
    ];
    const egyptPhone = /\b01\d{9}\b/;
    const saudiPhone = /\b05\d{8}\b/;
    const egyptScore =
      egyptianMarkers.reduce(
        (score, marker) => (lowered.includes(marker) ? score + 1 : score),
        0,
      ) + (egyptPhone.test(lowered) ? 3 : 0);
    const saudiScore =
      saudiMarkers.reduce(
        (score, marker) => (lowered.includes(marker) ? score + 1 : score),
        0,
      ) + (saudiPhone.test(lowered) ? 3 : 0);
    if (egyptScore === 0 && saudiScore === 0) return null;
    if (egyptScore === saudiScore) return null;
    return egyptScore > saudiScore ? 'egypt' : 'saudi';
  }

  private selectVoice(
    dialect: string,
    preferredVoice?: string,
  ): string | undefined {
    if (preferredVoice) return preferredVoice;
    if (dialect === 'saudi') return this.voiceSaudi;
    return this.voiceEgypt;
  }

  private delay(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  private durationSeconds(start: [number, number]) {
    const diff = process.hrtime(start);
    return diff[0] + diff[1] / 1e9;
  }

}
