import { Injectable, ExecutionContext } from '@nestjs/common';
import { ThrottlerGuard, ThrottlerException } from '@nestjs/throttler';
import { FastifyRequest } from 'fastify';

/**
 * Enhanced rate limiter with per-endpoint and per-tenant limits
 */
@Injectable()
export class EnhancedRateLimiterGuard extends ThrottlerGuard {
  /**
   * Endpoint-specific rate limits
   * More restrictive for expensive operations
   */
  private readonly endpointLimits: Record<string, { ttl: number; limit: number }> = {
    // ASR endpoints - expensive GPU operations
    '/asr/transcribe': { ttl: 60000, limit: 10 }, // 10 per minute
    '/asr/stream': { ttl: 60000, limit: 5 }, // 5 per minute
    
    // LLM endpoints - expensive GPU operations
    '/llm/infer': { ttl: 60000, limit: 20 }, // 20 per minute
    '/llm/chat': { ttl: 60000, limit: 30 }, // 30 per minute
    '/llm/orchestrate': { ttl: 60000, limit: 10 }, // 10 per minute
    
    // TTS endpoints
    '/tts/synthesize': { ttl: 60000, limit: 30 }, // 30 per minute
    
    // Auth endpoints - protect against brute force
    '/auth/login': { ttl: 60000, limit: 5 }, // 5 per minute
    '/auth/register': { ttl: 3600000, limit: 10 }, // 10 per hour
    '/auth/forgot-password': { ttl: 3600000, limit: 3 }, // 3 per hour
    
    // SOAP endpoints
    '/soap/generate': { ttl: 60000, limit: 20 }, // 20 per minute
    
    // FHIR endpoints
    '/fhir/push': { ttl: 60000, limit: 30 }, // 30 per minute
  };

  /**
   * Tenant tier rate limits (multipliers)
   */
  private readonly tierMultipliers: Record<string, number> = {
    free: 1,
    starter: 2,
    professional: 5,
    enterprise: 10,
  };

  protected async getTracker(req: FastifyRequest): Promise<string> {
    // Use a combination of IP and user ID for tracking
    const userId = (req as any).user?.id;
    const ip = this.getClientIp(req);
    
    if (userId) {
      return `user:${userId}`;
    }
    
    return `ip:${ip}`;
  }

  protected async handleRequest(
    context: ExecutionContext,
    limit: number,
    ttl: number,
  ): Promise<boolean> {
    const request = context.switchToHttp().getRequest<FastifyRequest>();
    const path = this.normalizePath(request.url);
    
    // Get endpoint-specific limits
    const endpointConfig = this.getEndpointConfig(path);
    
    // Get tenant tier multiplier
    const tierMultiplier = this.getTierMultiplier(request);
    
    // Calculate effective limits
    const effectiveLimit = endpointConfig
      ? Math.floor(endpointConfig.limit * tierMultiplier)
      : Math.floor(limit * tierMultiplier);
    
    const effectiveTtl = endpointConfig ? endpointConfig.ttl : ttl;

    // Call parent with adjusted limits
    const tracker = await this.getTracker(request);
    const key = this.generateKey(context, tracker, `${path}`);
    
    const { totalHits } = await this.storageService.increment(key, effectiveTtl);

    if (totalHits > effectiveLimit) {
      // Add rate limit headers
      const response = context.switchToHttp().getResponse();
      response.header('X-RateLimit-Limit', effectiveLimit.toString());
      response.header('X-RateLimit-Remaining', '0');
      response.header('X-RateLimit-Reset', Math.ceil(Date.now() / 1000 + effectiveTtl / 1000).toString());
      
      throw new ThrottlerException(`Rate limit exceeded. Maximum ${effectiveLimit} requests per ${effectiveTtl / 1000} seconds.`);
    }

    // Add rate limit headers for successful requests
    const response = context.switchToHttp().getResponse();
    response.header('X-RateLimit-Limit', effectiveLimit.toString());
    response.header('X-RateLimit-Remaining', Math.max(0, effectiveLimit - totalHits).toString());

    return true;
  }

  private getClientIp(request: FastifyRequest): string {
    const forwardedFor = request.headers['x-forwarded-for'];
    if (forwardedFor) {
      const ips = (forwardedFor as string).split(',');
      return ips[0].trim();
    }
    return request.ip || 'unknown';
  }

  private normalizePath(url: string): string {
    // Remove query parameters
    const path = url.split('?')[0];
    
    // Remove trailing slashes
    return path.replace(/\/+$/, '');
  }

  private getEndpointConfig(path: string): { ttl: number; limit: number } | undefined {
    // Exact match
    if (this.endpointLimits[path]) {
      return this.endpointLimits[path];
    }

    // Prefix match (for parameterized routes)
    for (const [endpoint, config] of Object.entries(this.endpointLimits)) {
      if (path.startsWith(endpoint)) {
        return config;
      }
    }

    return undefined;
  }

  private getTierMultiplier(request: FastifyRequest): number {
    const user = (request as any).user;
    const tier = user?.tier || user?.subscription || 'free';
    
    return this.tierMultipliers[tier] || 1;
  }
}
