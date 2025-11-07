// gateway/src/rag/rag.module.ts
/**
 * RAG Module
 * Week 5 Day 31
 */
import { Module } from '@nestjs/common';
import { RAGController } from './rag.controller';
import { VectorCacheService } from '../cache/vector-cache.service';

@Module({
  controllers: [RAGController],
  providers: [VectorCacheService],
  exports: [],
})
export class RAGModule {}
