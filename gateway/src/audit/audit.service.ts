import { Injectable, Logger } from '@nestjs/common';
import { Pool } from 'pg';

@Injectable()
export class AuditService {
  private readonly logger = new Logger(AuditService.name);
  private readonly pool: Pool | null;

  constructor() {
    const url = process.env.DATABASE_URL;
    this.pool = url ? new Pool({ connectionString: url }) : null;
  }

  async log(params: {
    actorId: string;
    action: string;
    resourceType: string;
    resourceId: string;
    metadata?: Record<string, any>;
  }): Promise<void> {
    if (!this.pool) {
      this.logger.warn('Audit pool not configured; skipping audit log');
      return;
    }
    const { actorId, action, resourceType, resourceId, metadata = {} } = params;
    try {
      await this.pool.query(
        `INSERT INTO audit_log (actor_id, action, resource_type, resource_id, metadata)
         VALUES ($1, $2, $3, $4, $5)`,
        [actorId, action, resourceType, resourceId, metadata],
      );
    } catch (err) {
      this.logger.warn(`Audit log failed for ${action}:${resourceId}`, { error: (err as Error).message });
    }
  }
}
