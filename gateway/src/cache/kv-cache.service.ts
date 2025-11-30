// gateway/src/cache/kv-cache.service.ts
/**
 * Key-Value Cache Service - Redis-based prompt caching
 * Week 3 Day 18 (Oct 12, 2025)
 */
import { Injectable, Logger } from '@nestjs/common';
import { createClient, RedisClientType } from 'redis';

@Injectable()
export class KvCacheService {
  private readonly logger = new Logger(KvCacheService.name);
  private readonly client: RedisClientType;
  private readonly DEFAULT_TTL = 3600; // 1 hour

  constructor() {
    this.client = createClient({
      url: `redis://${process.env.REDIS_HOST || 'localhost'}:${process.env.REDIS_PORT || 6379}`,
      database: 1, // Use separate DB for cache
    });

    this.client.on('error', (err) => {
      this.logger.error('Redis Cache Error', err);
    });

    this.client.connect().catch((err) => {
      this.logger.error('Failed to connect to Redis cache', err);
    });
  }

  /**
   * Set a cached value with optional TTL
   */
  async set(key: string, value: any, ttl?: number): Promise<void> {
    try {
      const serialized = JSON.stringify(value);
      await this.client.set(key, serialized, {
        EX: ttl || this.DEFAULT_TTL,
      });
      this.logger.debug(`Cached key: ${key}`);
    } catch (error) {
      this.logger.error(`Failed to cache key ${key}:`, error);
    }
  }

  /**
   * Get a cached value
   */
  async get<T = any>(key: string): Promise<T | null> {
    try {
      const value = await this.client.get(key);
      if (!value) {
        return null;
      }
      return JSON.parse(value) as T;
    } catch (error) {
      this.logger.error(`Failed to get cached key ${key}:`, error);
      return null;
    }
  }

  /**
   * Delete a cached value
   */
  async delete(key: string): Promise<void> {
    try {
      await this.client.del(key);
      this.logger.debug(`Deleted cached key: ${key}`);
    } catch (error) {
      this.logger.error(`Failed to delete key ${key}:`, error);
    }
  }

  /**
   * Check if key exists
   */
  async exists(key: string): Promise<boolean> {
    try {
      const result = await this.client.exists(key);
      return result === 1;
    } catch (error) {
      this.logger.error(`Failed to check key existence ${key}:`, error);
      return false;
    }
  }

  /**
   * Get or set pattern: if key exists, return it; otherwise compute and cache
   */
  async getOrSet<T>(
    key: string,
    factory: () => Promise<T>,
    ttl?: number,
  ): Promise<T> {
    const cached = await this.get<T>(key);
    if (cached !== null) {
      this.logger.debug(`Cache hit: ${key}`);
      return cached;
    }

    this.logger.debug(`Cache miss: ${key}, computing...`);
    const value = await factory();
    await this.set(key, value, ttl);
    return value;
  }

  /**
   * Clear all keys matching pattern
   */
  async clearPattern(pattern: string): Promise<void> {
    try {
      const keys = await this.client.keys(pattern);
      if (keys.length > 0) {
        await this.client.del(keys);
        this.logger.log(`Cleared ${keys.length} keys matching ${pattern}`);
      }
    } catch (error) {
      this.logger.error(`Failed to clear pattern ${pattern}:`, error);
    }
  }

  /**
   * Lightweight readiness check.
   */
  async ping(): Promise<boolean> {
    try {
      await this.client.ping();
      return true;
    } catch (error) {
      this.logger.warn('Redis ping failed', error);
      return false;
    }
  }

  /**
   * Cleanup on service shutdown
   */
  async onModuleDestroy() {
    await this.client.quit();
    this.logger.log('Redis cache client disconnected');
  }
}
