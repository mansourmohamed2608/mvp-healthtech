# Multi-Tenant Hardening Verification Guide

## PR-7 Security Hardening Checklist

This document provides verification commands to confirm tenant isolation is enforced.

---

## 1. AuditService tenantId Required

### TypeScript Verification
```bash
# Verify AuditService.log() requires tenantId at compile time
cd gateway
npx tsc --noEmit 2>&1 | grep -E "(audit|tenantId)"
# Should return NO errors related to audit or tenantId
```

### Grep Verification - All Audit Calls Include tenantId
```bash
# Search for audit calls - should ALL have tenantId parameter
grep -n "auditService.log" src/**/*.ts

# Verify tenantId is first parameter in all calls
grep -B1 -A8 "this.auditService.log" src/**/*.ts | grep -E "(tenantId:|actorId:)"
# Every auditService.log should show tenantId BEFORE actorId
```

### Unit Test
```bash
# Run AuditService unit tests
npx jest src/audit/audit.service.spec.ts --verbose
```

---

## 2. TenantGuard Production Hardening

### Verification - Header Rejected in Production
```bash
# Check TenantGuard rejects x-tenant-id header in production
grep -A20 "if (this.isProduction)" src/auth/tenant.guard.ts | head -30
# Should show: tenantId = jwtTenantId (JWT only)
# Should show: reject if headerTenantId && !jwtTenantId
```

### Integration Test
```bash
# Test in production mode - header should be rejected
NODE_ENV=production MULTI_TENANT=true \
  curl -X GET http://localhost:3001/api/soap/notes \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -H "x-tenant-id: malicious-tenant"
# Should return 403: "Production mode requires tenant_id in JWT claims"
```

---

## 3. TenantGuard Coverage on PHI Controllers

### Grep Verification - All PHI Controllers Protected
```bash
# List all controllers with JwtAuthGuard
grep -rn "@UseGuards(JwtAuthGuard" src/**/*.controller.ts

# List all controllers with TenantGuard
grep -rn "JwtAuthGuard, TenantGuard" src/**/*.controller.ts

# These should match (all JwtAuthGuard should also have TenantGuard):
# ✅ soap.controller.ts
# ✅ rag.controller.ts  
# ✅ session.controller.ts
# ✅ fhir.controller.ts
# ✅ conversation.controller.ts
# ✅ clinical.controller.ts
# ✅ va.controller.ts
# ✅ tts.controller.ts
# ✅ asr.controller.ts
# ✅ llm.controller.ts
# ✅ metrics.controller.ts
# ✅ twilio.controller.ts (token endpoint only)
```

### Expected TenantGuard Coverage
| Controller | Has TenantGuard | PHI Level |
|------------|-----------------|-----------|
| soap | ✅ Class-level | HIGH - SOAP notes |
| rag | ✅ Class-level | HIGH - Clinical context |
| session | ✅ Class-level | HIGH - Patient sessions |
| fhir | ✅ Class-level | HIGH - Medical records |
| conversation | ✅ Class-level | MEDIUM - Chat history |
| clinical | ✅ Class-level | HIGH - Clinical data |
| va | ✅ Class-level | MEDIUM - Appointments |
| tts | ✅ Class-level | LOW - Audio synthesis |
| asr | ✅ Class-level | MEDIUM - Transcripts |
| llm | ✅ Class-level | MEDIUM - LLM inference |
| metrics | ✅ Class-level | LOW - Aggregates only |
| twilio | ✅ Method-level | MEDIUM - Token generation |
| auth | ❌ N/A | N/A - Pre-auth endpoint |

---

## 4. RAG Purge Audit Logging

### Verification
```bash
# Check RAG purge has audit logging
grep -A15 "async purgeTenant" src/rag/rag.controller.ts | grep -E "(auditService|RAG_PURGE)"
# Should show: await this.auditService.log with action: 'RAG_PURGE'
```

### Test Purge Audit (Dev Mode)
```bash
# Purge with platform_admin role and verify audit log
curl -X DELETE "http://localhost:3001/api/rag/purge?tenantId=test-tenant" \
  -H "Authorization: Bearer $PLATFORM_ADMIN_TOKEN"

# Query audit_log table
psql $DATABASE_URL -c "SELECT * FROM audit_log WHERE action = 'RAG_PURGE' ORDER BY created_at DESC LIMIT 5;"
# Should show: tenant_id, actor_id, action='RAG_PURGE', deletedCount in metadata
```

---

## 5. CI Checks

### Required CI Checks
```yaml
# These checks should be in CI pipeline:
- name: Type Check
  run: cd gateway && npx tsc --noEmit

- name: Unit Tests (includes AuditService tests)
  run: cd gateway && npx jest --coverage

- name: TenantGuard Coverage Check
  run: |
    # Verify no JwtAuthGuard without TenantGuard on PHI controllers
    count_jwt=$(grep -c "@UseGuards(JwtAuthGuard)" gateway/src/**/*.controller.ts)
    count_tenant=$(grep -c "JwtAuthGuard, TenantGuard" gateway/src/**/*.controller.ts)
    # Allow auth.controller (no tenant pre-auth) and twilio (method-level)
    if [ $((count_jwt - count_tenant)) -gt 2 ]; then
      echo "FAIL: Some PHI controllers missing TenantGuard"
      exit 1
    fi
```

---

## 6. Database Verification

### Verify audit_log Has tenant_id
```sql
-- Check audit_log schema
\d audit_log
-- Should show: tenant_id VARCHAR(255) NOT NULL

-- Check no 'default' tenant entries in production
SELECT COUNT(*) FROM audit_log WHERE tenant_id = 'default';
-- In production: should be 0 (except 'system' for auth events)

-- Check all recent audit entries have proper tenant_id
SELECT tenant_id, action, COUNT(*) 
FROM audit_log 
WHERE created_at > NOW() - INTERVAL '1 day'
GROUP BY tenant_id, action
ORDER BY COUNT(*) DESC;
```

### Verify PHI Tables Have tenant_id
```sql
-- All PHI tables should have tenant_id column with NOT NULL constraint
SELECT table_name, column_name, is_nullable 
FROM information_schema.columns 
WHERE column_name = 'tenant_id' 
  AND table_schema = 'public'
ORDER BY table_name;
-- is_nullable should be 'NO' for all PHI tables
```

---

## 7. Quick Smoke Test

```bash
# 1. Start gateway in multi-tenant mode
MULTI_TENANT=true NODE_ENV=development npm run start:dev

# 2. Try to access PHI endpoint without tenant
curl -X GET http://localhost:3001/api/soap/notes \
  -H "Authorization: Bearer $JWT_WITHOUT_TENANT"
# Expected: 403 Forbidden - "Multi-tenant mode requires tenant_id claim"

# 3. Try with valid tenant in JWT
curl -X GET http://localhost:3001/api/soap/notes \
  -H "Authorization: Bearer $JWT_WITH_TENANT_CLAIM"
# Expected: 200 OK (or empty array)

# 4. Verify audit log entry was created with tenant_id
psql $DATABASE_URL -c "SELECT * FROM audit_log ORDER BY created_at DESC LIMIT 1;"
```

---

## Summary of Hardening Applied

| Security Control | Implementation | Verification |
|-----------------|----------------|--------------|
| AuditService.tenantId | Required (not optional) | TypeScript + Unit Test |
| TenantGuard Production | JWT claims only (no header) | Integration Test |
| PHI Controller Coverage | TenantGuard on all | Grep count match |
| RAG Purge Audit | Logs action + deletedCount | DB query |
| Runtime Validation | Throws if tenantId falsy | Unit Test |
| Compile-time Enforcement | TypeScript strict type | tsc --noEmit |
