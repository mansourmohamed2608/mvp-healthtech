import { Injectable, Logger, NotFoundException } from '@nestjs/common';
import { randomUUID } from 'crypto';
import { createClient, RedisClientType } from 'redis';
import { CreateSessionDto } from './dto/create-session.dto';
import { Session, SessionData } from './session.entity';
import {
  SessionResponseDto,
  CreateSessionResponseDto,
} from './dto/session-response.dto';

@Injectable()
export class SessionService {
  private readonly logger = new Logger(SessionService.name);
  private redisClient: RedisClientType | null = null;
  private readonly inMemoryStore = new Map<string, SessionData>();
  private readonly SESSION_TTL = 7200; // 2 hours in seconds
  private readonly SESSION_PREFIX = 'session:';
  private redisAvailable = false;

  constructor() {
    this.initializeRedis();
  }

  private initializeRedis() {
    const redisUrl = `redis://${process.env.REDIS_HOST || 'localhost'}:${process.env.REDIS_PORT || 6379}`;
    
    try {
      this.redisClient = createClient({
        url: redisUrl,
        password: process.env.REDIS_PASSWORD || undefined,
        socket: {
          connectTimeout: 2000,
          reconnectStrategy: () => false,
        },
      }) as RedisClientType;

      this.redisClient.on('error', (err) => {
        this.logger.warn('⚠️  Redis not available for sessions, using in-memory storage');
        this.redisAvailable = false;
      });

      this.redisClient.on('connect', () => {
        this.logger.log('✅ Redis connected for sessions');
        this.redisAvailable = true;
      });

      // Connect without blocking
      this.redisClient.connect().catch((error) => {
        this.logger.warn('⚠️  Redis connection failed for sessions, using in-memory storage');
        this.redisAvailable = false;
      });
    } catch (error) {
      this.logger.warn('⚠️  Redis initialization failed for sessions, using in-memory storage');
      this.redisAvailable = false;
    }
  }

  async create(dto: CreateSessionDto): Promise<CreateSessionResponseDto> {
    const sessionId = randomUUID();
    const now = new Date();
    const expiresAt = new Date(now.getTime() + this.SESSION_TTL * 1000);

    const session: SessionData = {
      sessionId,
      userId: dto.userId,
      callSid: dto.callSid,
      metadata: dto.metadata || {},
      createdAt: now.toISOString(),
      expiresAt: expiresAt.toISOString(),
      isActive: true,
    };

    // Use in-memory if Redis not available
    if (!this.redisAvailable || !this.redisClient) {
      this.inMemoryStore.set(sessionId, session);
      this.logger.log(`Session created (in-memory): ${sessionId}`);
      return {
        sessionId,
        issuedAt: now.toISOString(),
        expiresAt: expiresAt.toISOString(),
      };
    }

    // Use Redis if available
    try {
      await this.redisClient.setEx(
        `${this.SESSION_PREFIX}${sessionId}`,
        this.SESSION_TTL,
        JSON.stringify(session),
      );

      this.logger.log(`Session created: ${sessionId}`);

      return {
        sessionId,
        issuedAt: now.toISOString(),
        expiresAt: expiresAt.toISOString(),
      };
    } catch (error) {
      // Fallback to in-memory
      this.logger.warn('Redis error, using in-memory for session');
      this.inMemoryStore.set(sessionId, session);
      return {
        sessionId,
        issuedAt: now.toISOString(),
        expiresAt: expiresAt.toISOString(),
      };
    }
  }

  async findById(sessionId: string): Promise<SessionResponseDto> {
    // Check in-memory first
    if (!this.redisAvailable || !this.redisClient) {
      const session = this.inMemoryStore.get(sessionId);
      if (!session) {
        throw new NotFoundException(`Session ${sessionId} not found`);
      }
      return session;
    }

    // Use Redis
    try {
      const data = await this.redisClient.get(
        `${this.SESSION_PREFIX}${sessionId}`,
      );

      if (!data) {
        throw new NotFoundException(`Session ${sessionId} not found`);
      }

      const session: SessionData = JSON.parse(data);
      return session;
    } catch (error) {
      if (error instanceof NotFoundException) {
        throw error;
      }
      // Fallback to in-memory
      const session = this.inMemoryStore.get(sessionId);
      if (!session) {
        throw new NotFoundException(`Session ${sessionId} not found`);
      }
      return session;
    }
  }

  async update(
    sessionId: string,
    updates: Partial<SessionData>,
  ): Promise<SessionResponseDto> {
    try {
      const existing = await this.findById(sessionId);
      const updated: SessionData = {
        ...existing,
        ...updates,
        sessionId, // Prevent sessionId from being changed
      };

      const ttl = await this.redisClient.ttl(
        `${this.SESSION_PREFIX}${sessionId}`,
      );

      await this.redisClient.setEx(
        `${this.SESSION_PREFIX}${sessionId}`,
        ttl > 0 ? ttl : this.SESSION_TTL,
        JSON.stringify(updated),
      );

      this.logger.log(`Session updated: ${sessionId}`);
      return updated;
    } catch (error) {
      this.logger.error(`Failed to update session ${sessionId}`, error);
      throw error;
    }
  }

  async delete(sessionId: string): Promise<void> {
    try {
      const result = await this.redisClient.del(
        `${this.SESSION_PREFIX}${sessionId}`,
      );

      if (result === 0) {
        throw new NotFoundException(`Session ${sessionId} not found`);
      }

      this.logger.log(`Session deleted: ${sessionId}`);
    } catch (error) {
      if (error instanceof NotFoundException) {
        throw error;
      }
      this.logger.error(`Failed to delete session ${sessionId}`, error);
      throw error;
    }
  }

  async extendSession(sessionId: string): Promise<SessionResponseDto> {
    try {
      const session = await this.findById(sessionId);
      const newExpiresAt = new Date(
        Date.now() + this.SESSION_TTL * 1000,
      ).toISOString();

      await this.redisClient.expire(
        `${this.SESSION_PREFIX}${sessionId}`,
        this.SESSION_TTL,
      );

      const updated = await this.update(sessionId, {
        expiresAt: newExpiresAt,
      });

      this.logger.log(`Session extended: ${sessionId}`);
      return updated;
    } catch (error) {
      this.logger.error(`Failed to extend session ${sessionId}`, error);
      throw error;
    }
  }

  async isActive(sessionId: string): Promise<boolean> {
    try {
      const session = await this.findById(sessionId);
      return session.isActive;
    } catch (error) {
      if (error instanceof NotFoundException) {
        return false;
      }
      throw error;
    }
  }

  async onModuleDestroy() {
    if (this.redisClient) {
      await this.redisClient.quit();
    }
  }
}
