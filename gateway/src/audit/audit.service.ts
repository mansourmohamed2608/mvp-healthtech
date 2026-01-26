import { Injectable, Logger } from '@nestjs/common';
import { Pool } from 'pg';

/**
 * AuditService - Centralized audit logging for PHI operations
 * SECURITY: tenantId is REQUIRED to enforce tenant isolation in audit trail
 */
@Injectable()
export class AuditService {
  private readonly logger = new Logger(AuditService.name);
  private readonly pool: Pool | null;

  constructor() {
    const url = process.env.DATABASE_URL;
    this.pool = url ? new Pool({ connectionString: url }) : null;
  }

  /**
   * Log an auditable event
   * @param params - Audit log parameters
   * @param params.tenantId - REQUIRED: Tenant identifier for isolation
   * @throws Error if tenantId is missing (caught at compile time via type)
   */
  async log(params: {
    tenantId: string;  // REQUIRED - PR-7 hardening
    actorId: string;
    action: string;
    resourceType: string;
    resourceId: string;
    metadata?: Record<string, any>;
  }): Promise<void> {
    const { tenantId, actorId, action, resourceType, resourceId, metadata = {} } = params;

    // Runtime validation as defense-in-depth
    if (!tenantId) {
      this.logger.error(
        `SECURITY: Audit log called without tenantId for ${action}:${resourceId} by ${actorId}`,
      );
      throw new Error('tenantId is required for audit logging');
    }

    if (!this.pool) {
      this.logger.warn('Audit pool not configured; skipping audit log');
      return;
    }
    try {
      await this.pool.query(
        `INSERT INTO audit_log (actor_id, action, resource_type, resource_id, metadata, tenant_id)
         VALUES ($1, $2, $3, $4, $5, $6)`,
        [actorId, action, resourceType, resourceId, metadata, tenantId],
      );
    } catch (err) {
      this.logger.warn(`Audit log failed for ${action}:${resourceId}`, {
        error: (err as Error).message,
      });
    }
  }
}
