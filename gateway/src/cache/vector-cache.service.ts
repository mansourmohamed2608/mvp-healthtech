// gateway/src/cache/vector-cache.service.ts
/**
 * Vector Cache Service - In-memory vector store for few-shot examples
 * Week 3 Day 18 (Oct 12, 2025)
 * Uses simple in-memory storage, can be upgraded to Faiss/Qdrant later
 */
import { Injectable, Logger } from '@nestjs/common';

interface VectorEntry {
  id: string;
  vector: number[];
  metadata: Record<string, any>;
  text: string;
}

@Injectable()
export class VectorCacheService {
  private readonly logger = new Logger(VectorCacheService.name);
  private readonly cache: Map<string, VectorEntry> = new Map();
  private readonly MAX_ENTRIES = 1000;

  /**
   * Store a vector with metadata
   */
  async store(
    id: string,
    vector: number[],
    text: string,
    metadata: Record<string, any> = {},
  ): Promise<void> {
    // Implement LRU eviction if cache is full
    if (this.cache.size >= this.MAX_ENTRIES) {
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }

    this.cache.set(id, { id, vector, metadata, text });
    this.logger.debug(`Stored vector ${id}`);
  }

  /**
   * Find similar vectors using cosine similarity
   */
  async findSimilar(
    queryVector: number[],
    limit = 5,
    tenantId?: string,
  ): Promise<VectorEntry[]> {
    const results: Array<{ entry: VectorEntry; similarity: number }> = [];

    for (const entry of this.cache.values()) {
      if (tenantId && entry.metadata?.tenantId !== tenantId) {
        continue;
      }
      const similarity = this.cosineSimilarity(queryVector, entry.vector);
      results.push({ entry, similarity });
    }

    // Sort by similarity (descending) and return top k
    results.sort((a, b) => b.similarity - a.similarity);
    return results.slice(0, limit).map((r) => r.entry);
  }

  /**
   * Get vector by ID
   */
  async get(id: string): Promise<VectorEntry | null> {
    return this.cache.get(id) || null;
  }

  /**
   * Delete vector
   */
  async delete(id: string): Promise<void> {
    this.cache.delete(id);
    this.logger.debug(`Deleted vector ${id}`);
  }

  /**
   * Clear all vectors
   */
  async clear(): Promise<void> {
    this.cache.clear();
    this.logger.log('Cleared vector cache');
  }

  /**
   * Get cache statistics
   */
  getStats() {
    return {
      size: this.cache.size,
      maxSize: this.MAX_ENTRIES,
      utilizationPercent: (this.cache.size / this.MAX_ENTRIES) * 100,
    };
  }

  /**
   * Purge all vectors belonging to a specific tenant (PR-15 retention compliance)
   * @returns number of deleted entries
   */
  async purgeByTenant(tenantId: string): Promise<number> {
    let deleted = 0;
    for (const [key, entry] of this.cache.entries()) {
      if (entry.metadata?.tenantId === tenantId || key.startsWith(`${tenantId}:`)) {
        this.cache.delete(key);
        deleted++;
      }
    }
    this.logger.log(`Purged ${deleted} vectors for tenant ${tenantId}`);
    return deleted;
  }

  /**
   * Calculate cosine similarity between two vectors
   */
  private cosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length) {
      throw new Error('Vectors must have same dimensions');
    }

    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    normA = Math.sqrt(normA);
    normB = Math.sqrt(normB);

    if (normA === 0 || normB === 0) {
      return 0;
    }

    return dotProduct / (normA * normB);
  }
}
