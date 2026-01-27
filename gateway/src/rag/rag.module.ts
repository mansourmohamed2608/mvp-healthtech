// gateway/src/rag/rag.module.ts
/**
 * RAG Module
 * Week 5 Day 31
 */
import { Module } from '@nestjs/common';
import { RAGController } from './rag.controller';
import { VectorCacheService } from '../cache/vector-cache.service';
import { InternalHttpClient } from '../http/internal-http-client.service';
import { AuthModule } from '../auth/auth.module';

@Module({
  imports: [AuthModule],
  controllers: [RAGController],
  providers: [VectorCacheService, InternalHttpClient],
  exports: [],
})
export class RAGModule {}
