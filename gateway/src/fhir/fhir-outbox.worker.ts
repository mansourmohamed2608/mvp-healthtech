// gateway/src/fhir/fhir-outbox.worker.ts
/**
 * FhirOutboxWorker — delivers queued FHIR writes with exponential backoff.
 *
 * Pattern: transactional outbox.  soap.controller.ts writes to fhir_outbox
 * in the same DB transaction as the SOAP note status update.  This worker
 * polls every 30 seconds and pushes pending rows to the FHIR service.
 *
 * Retry schedule (attempt → delay before next retry):
 *   1 →  30 s   2 →  2 min   3 →  8 min   4 → 30 min
 *   5 →  2 h    6 →  8 h     7 → 24 h     8 → dead-letter
 */
import {
  Injectable,
  Logger,
  OnModuleInit,
  OnModuleDestroy,
  Inject,
} from '@nestjs/common';
import { Pool } from 'pg';
import axios from 'axios';
import { PG_POOL } from '../db/db.module';
import { AuditService } from '../audit/audit.service';

/** Exponential backoff delays in seconds (index = attempt number, 1-based) */
const BACKOFF_SECONDS = [0, 30, 120, 480, 1800, 7200, 28800, 86400];
const MAX_ATTEMPTS = BACKOFF_SECONDS.length; // after this → dead
const POLL_INTERVAL_MS = 30_000;
const BATCH_SIZE = 10;

@Injectable()
export class FhirOutboxWorker implements OnModuleInit, OnModuleDestroy {
  private readonly logger = new Logger(FhirOutboxWorker.name);
  private interval: ReturnType<typeof setInterval> | null = null;
  private readonly fhirBaseUrl: string;

  constructor(
    @Inject(PG_POOL) private readonly pool: Pool | null,
    private readonly auditService: AuditService,
  ) {
    this.fhirBaseUrl =
      process.env.FHIR_SERVICE_URL || 'http://localhost:5004';
  }

  onModuleInit() {
    if (!this.pool) {
      this.logger.warn('No DB pool — FHIR outbox worker disabled');
      return;
    }
    this.logger.log(
      `FHIR outbox worker started (poll every ${POLL_INTERVAL_MS / 1000}s)`,
    );
    // Run once immediately, then on interval
    void this.processOutbox();
    this.interval = setInterval(() => void this.processOutbox(), POLL_INTERVAL_MS);
  }

  onModuleDestroy() {
    if (this.interval) {
      clearInterval(this.interval);
      this.interval = null;
    }
  }

  private async processOutbox(): Promise<void> {
    if (!this.pool) return;
    let rows: any[];
    try {
      const result = await this.pool.query(
        `SELECT id, tenant_id, soap_note_id, idempotency_key, payload, attempts, max_attempts
         FROM fhir_outbox
         WHERE status IN ('pending', 'failed')
           AND (next_retry_at IS NULL OR next_retry_at <= now())
         ORDER BY next_retry_at NULLS FIRST
         LIMIT $1
         FOR UPDATE SKIP LOCKED`,
        [BATCH_SIZE],
      );
      rows = result.rows;
    } catch (err) {
      this.logger.error('Outbox poll query failed', (err as Error).message);
      return;
    }

    for (const row of rows) {
      await this.deliverRow(row);
    }
  }

  private async deliverRow(row: {
    id: string;
    tenant_id: string;
    soap_note_id: string;
    idempotency_key: string;
    payload: Record<string, unknown>;
    attempts: number;
    max_attempts: number;
  }): Promise<void> {
    const attempt = row.attempts + 1;
    try {
      const internalSecret = process.env.INTERNAL_SECRET || '';
      await axios.post(`${this.fhirBaseUrl}/fhir`, row.payload, {
        headers: {
          'Content-Type': 'application/json',
          'x-internal-secret': internalSecret,
          'x-idempotency-key': row.idempotency_key,
          'x-tenant-id': row.tenant_id,
        },
        timeout: 15_000,
      });

      // Success
      await this.pool!.query(
        `UPDATE fhir_outbox
         SET status = 'success', attempts = $1, processed_at = now(), last_error = null
         WHERE id = $2`,
        [attempt, row.id],
      );
      await this.pool!.query(
        `UPDATE soap_notes SET fhir_status = 'delivered', updated_at = now() WHERE id = $1`,
        [row.soap_note_id],
      );
      await this.auditService.log({
        tenantId: row.tenant_id,
        actorId: 'fhir-outbox-worker',
        action: 'FHIR_WRITE_DELIVERED',
        resourceType: 'soap_note',
        resourceId: row.soap_note_id,
        metadata: { attempt, idempotencyKey: row.idempotency_key },
      });
      this.logger.log(
        `Delivered FHIR outbox item ${row.id} (attempt ${attempt})`,
      );
    } catch (err) {
      const lastError = (err as Error).message;
      const isDead = attempt >= (row.max_attempts ?? MAX_ATTEMPTS);
      const backoffSec = BACKOFF_SECONDS[Math.min(attempt, BACKOFF_SECONDS.length - 1)];
      const nextRetry = new Date(Date.now() + backoffSec * 1000).toISOString();

      await this.pool!.query(
        `UPDATE fhir_outbox
         SET status        = $1,
             attempts      = $2,
             last_error    = $3,
             next_retry_at = $4
         WHERE id = $5`,
        [isDead ? 'dead_letter' : 'failed', attempt, lastError, nextRetry, row.id],
      );

      if (isDead) {
        this.logger.error(
          `FHIR outbox item ${row.id} moved to dead-letter after ${attempt} attempts: ${lastError}`,
        );
        await this.auditService.log({
          tenantId: row.tenant_id,
          actorId: 'fhir-outbox-worker',
          action: 'FHIR_WRITE_DEAD_LETTER',
          resourceType: 'soap_note',
          resourceId: row.soap_note_id,
          metadata: { attempt, lastError },
        });
      } else {
        this.logger.warn(
          `FHIR outbox item ${row.id} attempt ${attempt} failed (retry in ${backoffSec}s): ${lastError}`,
        );
      }
    }
  }
}
