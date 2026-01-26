// gateway/src/rag/rag.controller.ts
/**
 * RAG Controller - Endpoints for storing and retrieving knowledge
 * Week 5 Day 31 (Oct 25, 2025)
 */
import {
  Controller,
  Post,
  Get,
  Delete,
  Body,
  Query,
  UseGuards,
  Headers,
  ForbiddenException,
  Req,
} from '@nestjs/common';
import { VectorCacheService } from '../cache/vector-cache.service';
import { InternalHttpClient } from '../http/internal-http-client.service';
import { JwtAuthGuard } from '../auth/jwt.guard';
import { TenantGuard, getTenantId } from '../auth/tenant.guard';
import { Roles } from '../auth/roles.decorator';
import { camelResponse, wrapError } from '../utils/http-utils';
import { AuditService } from '../audit/audit.service';
import type { Request } from 'express';

interface StoreKnowledgeDto {
  key: string;
  text: string;
  metadata?: Record<string, any>;
  tenantId?: string;
}

interface SearchDto {
  query: string;
  limit?: number;
  tenantId?: string;
}

@UseGuards(JwtAuthGuard, TenantGuard)
@Roles('clinician')
@Controller('rag')
export class RAGController {
  private readonly llmClient;
  private readonly multiTenant = process.env.MULTI_TENANT === 'true';

  constructor(
    private readonly vectorCache: VectorCacheService,
    private readonly http: InternalHttpClient,
    private readonly auditService: AuditService,
  ) {
    const llmUrl = process.env.LLM_SERVICE_URL || 'http://localhost:5001';
    this.llmClient = this.http.getClient({
      baseUrl: llmUrl,
      serviceName: 'llm',
    });
  }

  /**
   * Resolve tenant ID from request, rejecting 'default' in multi-tenant mode
   */
  private resolveTenantId(req: Request, dtoTenantId?: string): string {
    // TenantGuard has already validated; use getTenantId for consistency
    const tenantId = getTenantId(req);
    // In multi-tenant mode, reject 'default' tenant for data isolation
    if (this.multiTenant && tenantId === 'default') {
      throw new ForbiddenException(
        'RAG operations require explicit tenant_id in multi-tenant mode',
      );
    }
    return tenantId;
  }

  @Post('store')
  async storeKnowledge(
    @Body() dto: StoreKnowledgeDto,
    @Req() req: Request,
    @Headers('x-tenant-id') tenantHeader?: string,
  ) {
    const tenantId = this.resolveTenantId(req, dto.tenantId);
    const vector = this.generateSimpleEmbedding(dto.text);

    await this.vectorCache.store(`${tenantId}:${dto.key}`, vector, dto.text, {
      ...(dto.metadata || {}),
      tenantId,
    });

    return camelResponse({ ok: true, key: dto.key, tenantId });
  }

  @Post('search')
  async searchSimilar(
    @Body() dto: SearchDto,
    @Req() req: Request,
    @Headers('x-tenant-id') tenantHeader?: string,
  ) {
    try {
      const tenantId = this.resolveTenantId(req, dto.tenantId);
      const queryVector = this.generateSimpleEmbedding(dto.query);
      const limit = dto.limit || 5;

      const results = await this.vectorCache.findSimilar(
        queryVector,
        limit,
        tenantId,
      );

      return camelResponse({
        query: dto.query,
        tenantId,
        results: results.map((r) => ({
          id: r.id,
          text: r.text,
          metadata: r.metadata,
        })),
      });
    } catch (error) {
      wrapError(error);
    }
  }

  @Post('note')
  async addClinicNote(
    @Body()
    dto: {
      title?: string;
      text: string;
      metadata?: Record<string, any>;
      tenantId?: string;
    },
    @Req() req: Request,
    @Headers('x-tenant-id') tenantHeader?: string,
  ) {
    const tenantId = this.resolveTenantId(req, dto.tenantId);
    const response = await this.llmClient.post(`/rag/note`, {
      ...dto,
      tenantId,
    });
    return camelResponse(response.data);
  }

  @Post('faq')
  async addFaq(
    @Body() dto: { question: string; answer: string; tenantId?: string },
    @Req() req: Request,
    @Headers('x-tenant-id') tenantHeader?: string,
  ) {
    const tenantId = this.resolveTenantId(req, dto.tenantId);
    const response = await this.llmClient.post(`/rag/faq`, {
      ...dto,
      tenantId,
    });
    return camelResponse(response.data);
  }

  @Get('notes')
  async listNotes(@Query('tenantId') tenantId?: string, @Req() req?: Request) {
    const resolvedTenantId = this.resolveTenantId(req as Request, tenantId);
    const response = await this.llmClient.get(`/rag/notes`, {
      params: { tenantId: resolvedTenantId },
    });
    return camelResponse(response.data);
  }

  @Get('stats')
  async getStats() {
    return this.vectorCache.getStats();
  }

  /**
   * Purge all vectors for a tenant (for retention compliance or tenant offboarding)
   * PR-15: Vector purge endpoint
   * Security: Only tenant's own admin OR platform_admin can purge
   */
  @Delete('purge')
  @Roles('admin', 'platform_admin')
  async purgeTenant(@Query('tenantId') tenantId: string, @Req() req: Request) {
    if (!tenantId || tenantId === 'default') {
      throw new ForbiddenException('Cannot purge default tenant');
    }
    // Verify admin belongs to this tenant OR is platform_admin
    const reqUser = (req as any).user;
    const userTenantId = getTenantId(req);
    const isPlatformAdmin = reqUser?.roles?.includes('platform_admin');
    if (!isPlatformAdmin && userTenantId !== tenantId) {
      throw new ForbiddenException(
        'Cannot purge another tenant\'s data without platform_admin role',
      );
    }
    const deleted = await this.vectorCache.purgeByTenant(tenantId);
    
    // Audit log for purge operation (PR-15 security hardening)
    const actor = reqUser?.sub || 'unknown';
    await this.auditService.log({
      tenantId,
      actorId: actor,
      action: 'RAG_PURGE',
      resourceType: 'rag_vectors',
      resourceId: tenantId,
      metadata: {
        deletedCount: deleted,
        isPlatformAdmin,
        requestingTenantId: userTenantId,
      },
    });
    
    return camelResponse({ ok: true, tenantId, deletedCount: deleted });
  }

  @Post('seed')
  async seedKnowledge() {
    // Seed with common medical knowledge
    const tenantId = 'default';
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
      await this.vectorCache.store(
        `${tenantId}:${item.key}`,
        vector,
        item.text,
        { ...(item.metadata || {}), tenantId },
      );
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
    const magnitude = Math.sqrt(
      vector.reduce((sum, val) => sum + val * val, 0),
    );
    if (magnitude > 0) {
      for (let i = 0; i < vector.length; i++) {
        vector[i] /= magnitude;
      }
    }

    return vector;
  }
}
