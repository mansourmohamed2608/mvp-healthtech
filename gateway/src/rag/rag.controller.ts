// gateway/src/rag/rag.controller.ts
/**
 * RAG Controller - Endpoints for storing and retrieving knowledge
 * Week 5 Day 31 (Oct 25, 2025)
 */
import { Controller, Post, Get, Body, Query, UseGuards } from '@nestjs/common';
import { VectorCacheService } from '../cache/vector-cache.service';
import { JwtAuthGuard } from '../auth/jwt.guard';
import { Roles } from '../auth/roles.decorator';
import { camelResponse, wrapError } from '../utils/http-utils';

interface StoreKnowledgeDto {
  key: string;
  text: string;
  metadata?: Record<string, any>;
}

interface SearchDto {
  query: string;
  limit?: number;
}

@UseGuards(JwtAuthGuard)
@Roles('clinician')
@Controller('rag')
export class RAGController {
  constructor(private readonly vectorCache: VectorCacheService) {}

  @Post('store')
  async storeKnowledge(@Body() dto: StoreKnowledgeDto) {
    // Generate simple embedding (in production, use actual embedding model)
    const vector = this.generateSimpleEmbedding(dto.text);
    
    await this.vectorCache.store(dto.key, vector, dto.text, dto.metadata || {});
    
    return camelResponse({ ok: true, key: dto.key });
  }

  @Post('search')
  async searchSimilar(@Body() dto: SearchDto) {
    try {
      const queryVector = this.generateSimpleEmbedding(dto.query);
      const limit = dto.limit || 5;
      
      const results = await this.vectorCache.findSimilar(queryVector, limit);
      
      return camelResponse({
        query: dto.query,
        results: results.map(r => ({
          id: r.id,
          text: r.text,
          metadata: r.metadata,
        })),
      });
    } catch (error) {
      wrapError(error);
    }
  }

  @Get('stats')
  async getStats() {
    return this.vectorCache.getStats();
  }

  @Post('seed')
  async seedKnowledge() {
    // Seed with common medical knowledge
    const knowledge = [
      {
        key: 'fever_management',
        text: 'إدارة الحمى: الراحة، شرب السوائل، خافضات الحرارة مثل الباراسيتامول أو الإيبوبروفين',
        metadata: { type: 'treatment', condition: 'fever' },
      },
      {
        key: 'covid_symptoms',
        text: 'أعراض كوفيد-19: حمى، سعال جاف، تعب، فقدان حاسة الشم أو التذوق، ضيق التنفس',
        metadata: { type: 'symptoms', condition: 'covid19' },
      },
      {
        key: 'chest_pain_emergency',
        text: 'ألم الصدر مع ضيق التنفس أو التعرق أو الغثيان يتطلب عناية طبية فورية - قد يكون نوبة قلبية',
        metadata: { type: 'emergency', condition: 'chest_pain' },
      },
      {
        key: 'diabetes_diet',
        text: 'نظام غذائي لمرضى السكري: تجنب السكريات البسيطة، تناول الحبوب الكاملة، الخضروات، البروتينات الخالية من الدهون',
        metadata: { type: 'lifestyle', condition: 'diabetes' },
      },
      {
        key: 'asthma_triggers',
        text: 'محفزات الربو الشائعة: الغبار، العفن، الدخان، التمارين الشديدة، الهواء البارد، الحساسية',
        metadata: { type: 'prevention', condition: 'asthma' },
      },
    ];

    for (const item of knowledge) {
      const vector = this.generateSimpleEmbedding(item.text);
      await this.vectorCache.store(item.key, vector, item.text, item.metadata);
    }

    return { ok: true, seeded: knowledge.length };
  }

  /**
   * Generate simple embedding based on character frequencies
   * In production, use actual embedding model (e.g., sentence-transformers)
   */
  private generateSimpleEmbedding(text: string): number[] {
    // Simple embedding: 128-dim vector based on character distribution
    const vector = new Array(128).fill(0);
    
    for (let i = 0; i < text.length; i++) {
      const charCode = text.charCodeAt(i);
      const index = charCode % 128;
      vector[index] += 1;
    }
    
    // Normalize
    const magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
    if (magnitude > 0) {
      for (let i = 0; i < vector.length; i++) {
        vector[i] /= magnitude;
      }
    }
    
    return vector;
  }
}
