// gateway/src/db/tenant-db.service.ts
/**
 * TenantDbService — pool wrapper that injects `SET LOCAL app.tenant_id`
 * before every query so PostgreSQL RLS policies (migration 005) can filter
 * rows to the correct tenant without any change to individual query strings.
 *
 * Usage:
 *   constructor(@Inject(TENANT_DB) private readonly db: TenantDbService) {}
 *
 *   // Single query:
 *   const rows = await this.db.query(tenantId, 'SELECT * FROM soap_notes');
 *
 *   // Transaction:
 *   await this.db.withTransaction(tenantId, async (client) => {
 *     await client.query('INSERT INTO soap_notes ...');
 *     await client.query('INSERT INTO audit_log ...');
 *   });
 */
import { Injectable, Inject, Logger } from '@nestjs/common';
import { Pool, PoolClient, QueryResultRow } from 'pg';
import { PG_POOL } from './db.module';

export const TENANT_DB = 'TENANT_DB';

@Injectable()
export class TenantDbService {
  private readonly logger = new Logger(TenantDbService.name);

  constructor(@Inject(PG_POOL) private readonly pool: Pool | null) {}

  /**
   * Execute a single SQL statement in tenant scope.
   * Acquires a client, sets app.tenant_id, runs the query, releases.
   */
  async query<T extends QueryResultRow = QueryResultRow>(
    tenantId: string,
    text: string,
    values?: unknown[],
  ): Promise<T[]> {
    if (!this.pool) {
      this.logger.warn('TenantDbService.query called but pool is null');
      return [];
    }
    const client = await this.pool.connect();
    try {
      // SET LOCAL is transaction-scoped; wrap in implicit transaction for safety
      await client.query('BEGIN');
      await client.query('SET LOCAL app.tenant_id = $1', [tenantId]);
      const result = await client.query<T>(text, values);
      await client.query('COMMIT');
      return result.rows;
    } catch (err) {
      await client.query('ROLLBACK').catch(() => undefined);
      throw err;
    } finally {
      client.release();
    }
  }

  /**
   * Execute multiple statements in a single tenant-scoped transaction.
   * The supplied callback receives a connected PoolClient with app.tenant_id
   * already set; BEGIN and COMMIT/ROLLBACK are managed by this method.
   */
  async withTransaction<T>(
    tenantId: string,
    fn: (client: PoolClient) => Promise<T>,
  ): Promise<T> {
    if (!this.pool) throw new Error('DB pool not available');
    const client = await this.pool.connect();
    try {
      await client.query('BEGIN');
      await client.query('SET LOCAL app.tenant_id = $1', [tenantId]);
      const result = await fn(client);
      await client.query('COMMIT');
      return result;
    } catch (err) {
      await client.query('ROLLBACK').catch(() => undefined);
      throw err;
    } finally {
      client.release();
    }
  }
}
