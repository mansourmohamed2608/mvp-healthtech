# Multi-Clinic Blockers Implementation

> **Date**: Implementation ready for review  
> **PRs**: PR-6, PR-7, PR-8, PR-15

---

## PR-6: Tenant Isolation End-to-End

### Overview
Adds `tenant_id` column to ALL PHI tables with indexes for query performance.

### Files Changed
| File | Change |
|------|--------|
| `infra/db/migrations/001_add_tenant_id.sql` | **NEW** - Migration adding tenant_id to 12+ PHI tables |

### Migration SQL Highlights
```sql
-- Adds tenant_id to all PHI tables:
-- sessions, soap_notes, appointments, patients, patient_documents,
-- patient_rag_items, conversation_messages, audit_log, soap_jobs,
-- doctors, doctor_schedules, clinicians, soap_note_audit

ALTER TABLE sessions ADD COLUMN IF NOT EXISTS tenant_id text NOT NULL DEFAULT 'default';
CREATE INDEX IF NOT EXISTS idx_sessions_tenant ON sessions(tenant_id);
-- ... repeated for all tables with composite indexes where appropriate
```

### Verify Steps
```powershell
# 1. Check migration file exists
Get-Content mvp-healthtech/infra/db/migrations/001_add_tenant_id.sql | Select-String "tenant_id"

# 2. Dry-run migration (local Postgres)
psql $DATABASE_URL -f infra/db/migrations/001_add_tenant_id.sql --set ON_ERROR_STOP=1

# 3. Verify columns added
psql $DATABASE_URL -c "\d soap_notes" | Select-String "tenant_id"
```

---

## PR-7: Tenant Enforcement

### Overview
If `MULTI_TENANT=true`, reject requests missing tenant claim with 403 Forbidden.

### Files Changed
| File | Change |
|------|--------|
| `gateway/src/auth/tenant.guard.ts` | **NEW** - TenantGuard + getTenantId helper |
| `gateway/src/soap/soap.controller.ts` | Import TenantGuard, add to @UseGuards |
| `gateway/src/rag/rag.controller.ts` | Import TenantGuard, add to @UseGuards |

### Code Edits

**tenant.guard.ts** (new file):
```typescript
@Injectable()
export class TenantGuard implements CanActivate {
  canActivate(context: ExecutionContext): boolean {
    if (process.env.MULTI_TENANT !== 'true') return true;
    
    const request = context.switchToHttp().getRequest<Request>();
    const tenantId = getTenantId(request);
    
    if (!tenantId) {
      throw new ForbiddenException('tenant_id required in multi-tenant mode');
    }
    return true;
  }
}

export function getTenantId(req: Request): string {
  return (req as any).user?.tenant_id 
    || req.headers['x-tenant-id'] as string 
    || 'default';
}
```

**soap.controller.ts** changes:
```typescript
// Added imports
import { TenantGuard, getTenantId } from '../auth/tenant.guard';
import { ForbiddenException } from '@nestjs/common';
import { randomUUID } from 'crypto';

// Changed decorator
@UseGuards(JwtAuthGuard, TenantGuard)  // was: @UseGuards(JwtAuthGuard)
@Controller('soap')
```

### Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `MULTI_TENANT` | `false` | Enable multi-tenant enforcement |
| `ENABLE_DEV_TENANT_FALLBACK` | `false` | Allow 'default' tenant in dev (non-prod only) |

### Verify Steps
```powershell
# 1. Check guard exports
Select-String -Path gateway/src/auth/tenant.guard.ts -Pattern "export.*TenantGuard"

# 2. Check controllers use guard
Select-String -Path gateway/src/soap/soap.controller.ts -Pattern "TenantGuard"
Select-String -Path gateway/src/rag/rag.controller.ts -Pattern "TenantGuard"

# 3. Test rejection (requires running gateway)
# Without tenant header in multi-tenant mode:
curl -X POST http://localhost:3000/soap/generate -H "Authorization: Bearer $TOKEN" -d '{}'
# Expected: 403 Forbidden "tenant_id required in multi-tenant mode"
```

---

## PR-8: FHIR Writeback Integrity

### Overview
- Separate `approval_status` from `fhir_status`
- Transactional outbox pattern for reliable delivery
- Exponential backoff retry (1min, 5min, 30min)
- Idempotency keys prevent duplicates
- Daily reconciliation catches orphaned writes

### Files Changed
| File | Change |
|------|--------|
| `infra/db/migrations/002_add_fhir_status.sql` | **NEW** - fhir_status column, fhir_outbox + fhir_reconciliation tables |
| `services/fhir/outbox_worker.py` | **NEW** - Worker script with retry logic |
| `infra/k8s/fhir-outbox-worker.yaml` | **NEW** - K8s CronJob for worker + reconciliation |
| `gateway/src/soap/soap.controller.ts` | Modified approve endpoint to enqueue to outbox |

### Migration SQL Highlights
```sql
-- New columns on soap_notes
ALTER TABLE soap_notes ADD COLUMN fhir_status text DEFAULT 'not_requested';
ALTER TABLE soap_notes ADD COLUMN fhir_idempotency_key text;
ALTER TABLE soap_notes ADD COLUMN fhir_resource_id text;

-- Outbox table
CREATE TABLE IF NOT EXISTS fhir_outbox (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id text NOT NULL DEFAULT 'default',
  soap_note_id uuid REFERENCES soap_notes(id),
  idempotency_key text UNIQUE NOT NULL,
  payload jsonb NOT NULL,
  status text DEFAULT 'pending',  -- pending, processing, success, failed, dead_letter
  attempts integer DEFAULT 0,
  max_attempts integer DEFAULT 3,
  next_retry_at timestamptz DEFAULT NOW()
);

-- Reconciliation tracking
CREATE TABLE IF NOT EXISTS fhir_reconciliation (
  id serial PRIMARY KEY,
  run_at timestamptz DEFAULT NOW(),
  orphaned_found integer DEFAULT 0,
  re_enqueued integer DEFAULT 0,
  status text DEFAULT 'running'
);
```

### Approve Endpoint Changes
```typescript
// When FHIR_OUTBOX_ENABLED=true, enqueue instead of direct call
await this.pool.query('BEGIN');
await this.pool.query(
  `UPDATE soap_notes SET fhir_status = 'pending', fhir_idempotency_key = $1 WHERE id = $2 AND tenant_id = $3`,
  [idempotencyKey, noteId, tenantId]
);
await this.pool.query(
  `INSERT INTO fhir_outbox (tenant_id, soap_note_id, idempotency_key, payload, status, next_retry_at)
   VALUES ($1, $2, $3, $4, 'pending', NOW())`,
  [tenantId, noteId, idempotencyKey, JSON.stringify(payload)]
);
await this.pool.query('COMMIT');
```

### Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `FHIR_OUTBOX_ENABLED` | `false` | Use outbox pattern instead of direct FHIR call |
| `OUTBOX_BATCH_SIZE` | `50` | Items per worker run |

### Verify Steps
```powershell
# 1. Check migration
Get-Content mvp-healthtech/infra/db/migrations/002_add_fhir_status.sql | Select-String "fhir_outbox"

# 2. Check outbox worker
Select-String -Path services/fhir/outbox_worker.py -Pattern "def process_item"

# 3. Check K8s CronJob
Get-Content infra/k8s/fhir-outbox-worker.yaml | Select-String "schedule:"

# 4. Verify approve endpoint uses outbox
Select-String -Path gateway/src/soap/soap.controller.ts -Pattern "fhir_outbox"

# 5. Test outbox flow (requires DB)
psql $DATABASE_URL -c "SELECT * FROM fhir_outbox LIMIT 5;"
```

---

## PR-15: RAG/Qdrant Tenant Isolation

### Overview
- Enforce tenant filter on EVERY vector query
- Reject 'default' tenant in multi-tenant mode
- Vector purge endpoint for retention compliance

### Files Changed
| File | Change |
|------|--------|
| `gateway/src/rag/rag.controller.ts` | Strict tenant enforcement + purge endpoint |
| `gateway/src/cache/vector-cache.service.ts` | Added `purgeByTenant()` method |

### Code Edits

**rag.controller.ts** changes:
```typescript
// Added imports
import { TenantGuard, getTenantId } from '../auth/tenant.guard';
import { ForbiddenException, Delete, Req } from '@nestjs/common';

// Added guard
@UseGuards(JwtAuthGuard, TenantGuard)

// New method to reject 'default' in multi-tenant
private resolveTenantId(req: Request, dtoTenantId?: string): string {
  const tenantId = getTenantId(req);
  if (this.multiTenant && tenantId === 'default') {
    throw new ForbiddenException(
      'RAG operations require explicit tenant_id in multi-tenant mode'
    );
  }
  return tenantId;
}

// New purge endpoint
@Delete('purge')
@Roles('admin')
async purgeTenant(@Query('tenantId') tenantId: string) {
  if (!tenantId || tenantId === 'default') {
    throw new ForbiddenException('Cannot purge default tenant');
  }
  const deleted = await this.vectorCache.purgeByTenant(tenantId);
  return { ok: true, tenantId, deletedCount: deleted };
}
```

**vector-cache.service.ts** addition:
```typescript
async purgeByTenant(tenantId: string): Promise<number> {
  let deleted = 0;
  for (const [key, entry] of this.cache.entries()) {
    if (entry.metadata?.tenantId === tenantId || key.startsWith(`${tenantId}:`)) {
      this.cache.delete(key);
      deleted++;
    }
  }
  this.logger.log(`Purged ${deleted} vectors for tenant ${tenantId}`);
  return deleted;
}
```

### Verify Steps
```powershell
# 1. Check RAG controller has strict enforcement
Select-String -Path gateway/src/rag/rag.controller.ts -Pattern "resolveTenantId"

# 2. Check purge endpoint exists
Select-String -Path gateway/src/rag/rag.controller.ts -Pattern "purgeTenant"

# 3. Check vector cache has purge method
Select-String -Path gateway/src/cache/vector-cache.service.ts -Pattern "purgeByTenant"

# 4. Test rejection in multi-tenant mode
# MULTI_TENANT=true, no tenant header:
curl -X POST http://localhost:3000/rag/search -H "Authorization: Bearer $TOKEN" -d '{"query":"test"}'
# Expected: 403 "RAG operations require explicit tenant_id in multi-tenant mode"
```

---

## Deployment Checklist

### Pre-deployment
- [ ] Review all migration SQL files
- [ ] Backup database before running migrations
- [ ] Set `MULTI_TENANT=false` initially (gradual rollout)
- [ ] Set `FHIR_OUTBOX_ENABLED=false` initially

### Migration Order
1. Run `001_add_tenant_id.sql` - Adds columns (safe, defaults to 'default')
2. Run `002_add_fhir_status.sql` - Adds FHIR tracking tables
3. Deploy updated gateway code
4. Deploy outbox worker CronJob

### Post-deployment Verification
```powershell
# Check all tables have tenant_id
psql $DATABASE_URL -c "
  SELECT table_name, column_name 
  FROM information_schema.columns 
  WHERE column_name = 'tenant_id' 
  ORDER BY table_name;
"

# Check outbox is processing
psql $DATABASE_URL -c "SELECT status, COUNT(*) FROM fhir_outbox GROUP BY status;"

# Check vector cache stats
curl http://localhost:3000/rag/stats
```

### Gradual Rollout
1. **Week 1**: Deploy with `MULTI_TENANT=false`, `FHIR_OUTBOX_ENABLED=false`
2. **Week 2**: Enable `FHIR_OUTBOX_ENABLED=true`, monitor outbox processing
3. **Week 3**: Enable `MULTI_TENANT=true` for pilot clinics
4. **Week 4**: Full multi-tenant enforcement

---

## Test Matrix

| Scenario | PR | Test Command |
|----------|-----|--------------|
| Tenant column exists | PR-6 | `psql -c "\d soap_notes"` |
| Missing tenant rejected | PR-7 | `curl -X POST /soap/generate` (no header, MULTI_TENANT=true) |
| Outbox enqueues on approve | PR-8 | Check `fhir_outbox` table after approve |
| Outbox worker processes | PR-8 | Run `python outbox_worker.py`, check `fhir_status='success'` |
| RAG rejects default tenant | PR-15 | `curl -X POST /rag/search` (MULTI_TENANT=true) |
| Vector purge works | PR-15 | `curl -X DELETE /rag/purge?tenantId=test-tenant` |
