// gateway/src/auth/tenant.guard.ts
/**
 * Tenant Guard - Enforces tenant isolation in multi-tenant mode
 * PR-7: Reject requests missing tenant claim when MULTI_TENANT=true
 * SECURITY HARDENING: Production accepts tenant ONLY from JWT claims (never headers)
 */
import {
  Injectable,
  CanActivate,
  ExecutionContext,
  ForbiddenException,
  Logger,
} from '@nestjs/common';
import type { Request } from 'express';

interface RequestWithTenant extends Request {
  tenantId?: string;
  user?: {
    sub?: string;
    tenant_id?: string;
    tenantId?: string;
  };
}

@Injectable()
export class TenantGuard implements CanActivate {
  private readonly logger = new Logger(TenantGuard.name);
  private readonly multiTenantEnabled =
    process.env.MULTI_TENANT === 'true' || process.env.MULTI_TENANT === '1';
  private readonly isProduction = process.env.NODE_ENV === 'production';
  private readonly devFallbackEnabled =
    (process.env.ENABLE_DEV_TENANT_FALLBACK === 'true' ||
      process.env.ENABLE_DEV_TENANT_FALLBACK === '1') &&
    !this.isProduction;

  canActivate(context: ExecutionContext): boolean {
    // If multi-tenant is not enabled, allow all requests
    if (!this.multiTenantEnabled) {
      return true;
    }

    const request = context.switchToHttp().getRequest<RequestWithTenant>();
    const user = request.user;

    // SECURITY: In production, tenant MUST come from JWT claims only
    // x-tenant-id header is allowed ONLY in non-production environments
    let tenantId: string | undefined;
    const jwtTenantId = user?.tenant_id || user?.tenantId;
    
    if (this.isProduction) {
      // Production: JWT claims ONLY
      tenantId = jwtTenantId;
      
      // Log and reject if someone tries to use header in production
      const headerTenantId = request.headers['x-tenant-id'];
      if (headerTenantId && !jwtTenantId) {
        this.logger.warn(
          `SECURITY: Rejected x-tenant-id header in production (user: ${user?.sub || 'unknown'})`,
        );
        throw new ForbiddenException(
          'Production mode requires tenant_id in JWT claims. Header override is disabled.',
        );
      }
    } else {
      // Non-production: Allow JWT claims or header fallback
      tenantId = jwtTenantId || (request.headers['x-tenant-id'] as string | undefined);
    }

    if (!tenantId) {
      // Dev fallback: only in non-production with explicit opt-in
      if (this.devFallbackEnabled) {
        this.logger.warn(
          'Request missing tenant_id, using dev fallback tenant "default"',
        );
        request.tenantId = 'default';
        return true;
      }

      this.logger.warn(
        `Rejected request: missing tenant_id claim (user: ${user?.sub || 'unknown'})`,
      );
      throw new ForbiddenException(
        this.isProduction
          ? 'Multi-tenant mode requires tenant_id claim in JWT'
          : 'Multi-tenant mode requires tenant_id claim in JWT or x-tenant-id header',
      );
    }

    // Reject 'default' tenant in production multi-tenant mode
    if (this.isProduction && tenantId === 'default') {
      this.logger.warn(
        `Rejected 'default' tenant in production (user: ${user?.sub || 'unknown'})`,
      );
      throw new ForbiddenException(
        "'default' tenant is not allowed in production multi-tenant mode",
      );
    }

    // Attach tenant ID to request for downstream use
    request.tenantId = tenantId;
    return true;
  }
}

/**
 * Helper to extract tenant ID from request
 * Use in controllers/services: const tenantId = getTenantId(req);
 * SECURITY: In production, only accepts JWT claims (not headers)
 */
export function getTenantId(request: Request): string {
  const multiTenantEnabled =
    process.env.MULTI_TENANT === 'true' || process.env.MULTI_TENANT === '1';
  const isProduction = process.env.NODE_ENV === 'production';

  const req = request as RequestWithTenant;

  // From TenantGuard (already validated)
  if (req.tenantId) {
    return req.tenantId;
  }

  // From JWT
  const user = req.user;
  const tenantFromJwt = user?.tenant_id || user?.tenantId;
  if (tenantFromJwt) {
    return tenantFromJwt;
  }

  // From header - ONLY in non-production
  if (!isProduction) {
    const tenantFromHeader = request.headers['x-tenant-id'];
    if (typeof tenantFromHeader === 'string') {
      return tenantFromHeader;
    }
  }

  // Default for single-tenant mode
  if (!multiTenantEnabled) {
    return 'default';
  }

  // Should not reach here if TenantGuard is used
  throw new ForbiddenException('tenant_id required in multi-tenant mode');
}
