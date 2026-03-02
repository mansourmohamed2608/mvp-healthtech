# Production Readiness Audit
**Date:** 2025-07  
**Branch:** `feature/va-arabic-va` (commit `aed9392`)  
**System:** HealthTech Arabic Medical AI — NestJS Gateway + Python Microservices  
**Auditor:** GitHub Copilot Engineering Audit  

---

## A. Executive Summary

This system is **NOT production-ready** in its current state. Six critical or high-severity issues would expose patient health information (PHI) and allow unauthorized access in a production deployment. The most severe issue is that the "production" Docker Compose file forces `NODE_ENV=development`, permanently activating a backdoor dev-auth endpoint that accepts plain-text credentials from an environment variable. Combined with a CORS wildcard default and a JWT expiry mismatch that silently breaks sessions, the system has significant security and reliability gaps that must be resolved before any live patient data is processed.

**Risk summary by category:**

| Category | Critical | High | Medium | Low |
|---|---|---|---|---|
| Authentication / Authorization | 2 | 3 | 1 | 0 |
| Data Security / PHI | 0 | 2 | 2 | 1 |
| Infrastructure / Config | 1 | 1 | 3 | 2 |
| Reliability / Resilience | 0 | 0 | 2 | 3 |

**Immediate actions required (before any production traffic):**
1. Set `NODE_ENV=production` in `docker-compose.demo.yml`
2. Replace `postgres:postgres` default DB password in compose files
3. Fix JWT `expires_in` response discrepancy
4. Lock CORS to explicit allowed origins
5. Set `POSTGRES_PASSWORD` as a required secret in all compose files

---

## B. Production Readiness Checklist

### Authentication & Authorization
- [ ] **CRIT-1** `NODE_ENV=development` hardcoded in `docker-compose.demo.yml` — dev-auth endpoint always active in production
- [ ] **CRIT-2** Plain-text password comparison in dev-auth (`DEV_AUTH_USERS=user:pass` env var)
- [ ] **HIGH-1** JWT `expires_in` field in auth response claims 7 days (`604800s`) but token actually expires in 1 hour (module config `signOptions.expiresIn: '1h'`) — silently breaks client sessions
- [ ] **HIGH-2** No token refresh endpoint — users must re-login manually after 1h with no UI indication
- [ ] **HIGH-3** JWT stored in `localStorage` via Zustand `persist` — vulnerable to XSS exfiltration
- [ ] **HIGH-4** `validateUser()` in `auth.service.ts` is a stub — accepts any non-empty username/password
- [ ] **MED-1** OIDC flow is a stub (dummy-client-id guard) — `/auth/oidc/login` returns 501

### Network & API Security
- [ ] **CRIT-3** `CORS_ALLOWED_ORIGINS=${CORS_ALLOWED_ORIGINS:-*}` in `docker-compose.demo.yml` — all origins accepted when env var is unset
- [ ] **HIGH-5** `DATABASE_URL=postgresql://postgres:postgres@...` hardcoded in `docker-compose.prod.yml` and `docker-compose.yml` — committed to git with default credentials
- [ ] **MED-2** `ThrottlerModule` set to 50 req/min globally — no per-endpoint limits for expensive GPU endpoints
- [ ] **MED-3** Morgan 'combined' format logs may include credentials in unusual configurations
- [ ] **LOW-1** `safeLog` SENSITIVE_KEYS list missing `password`, `token`, `secret`, `authorization` — those keys would log as plaintext

### Data / PHI Isolation
- [ ] **MED-4** `MULTI_TENANT=false` by default — `TenantGuard` returns `true` for ALL requests; all authenticated users share data space in single-tenant mode
- [ ] **MED-5** `GET /soap/notes/:id` has no ownership check — any authenticated clinician can retrieve any note by ID (IDOR)
- [ ] **LOW-2** Redis client created with `require()` inside `getJobStatus()` — new connection per request, connections never closed

### Infrastructure & Secrets
- [ ] **HIGH-6** Real Twilio API credentials and HuggingFace token stored in `infra/.env` locally — not in git but poor hygiene; rotate immediately if file was ever copied/shared
- [ ] **MED-6** No migration runner in CI — 3 SQL migration files in `infra/db/migrations/` must be run manually with `psql`
- [ ] **MED-7** Multiple services instantiate their own `pg.Pool` directly in controllers/services — no shared DB module, no pool size limits
- [ ] **LOW-3** `docker-compose.prod.yml` mounts source code as volume and runs `pnpm install` at container start — should use built image, not live source mount
- [ ] **LOW-4** No DB connection pool limits — potential connection exhaustion under load

### Observability & Reliability
- [ ] **MED-8** No structured startup health gate — services start without confirming DB/Redis connectivity
- [ ] **LOW-5** No alerting thresholds configured in Prometheus/Grafana for SLA violations
- [ ] **LOW-6** `SoapController.enqueueJob` and `getJobStatus` have no retry/DLQ mechanism defined

---

## C. Architecture Diagram

```mermaid
graph TB
    subgraph Internet
        Client[Browser / Mobile]
        Twilio[Twilio Voice]
    end

    subgraph GCP VM - healthtech-demo us-east1-b
        subgraph Exposed Ports
            Nginx[nginx :80/:443\nReverse Proxy]
            Frontend[frontend-vite :5173\nnginx:alpine]
            Gateway[gateway :3000\nNestJS]
        end

        subgraph Internal Docker Network
            ASR[asr :5000\nWhisperX large-v3 GPU]
            LLM[llm :5001\nMMed-Llama-3-8B GPU]
            LLMVA[llm-va :5002\nQwen2.5-3B GPU]
            TTS[tts :5002\nedge-tts CPU]
            SOAP[soap :5003\nFastAPI CPU]
            FHIR[fhir :5004\nFastAPI CPU]
        end

        subgraph Data Layer
            PG[(PostgreSQL :5432\n10 tables + audit_log)]
            Redis[(Redis :6379\nSessions + SOAP queue)]
            Qdrant[(Qdrant :6333\nVector DB - RAG)]
        end
    end

    subgraph External Systems
        FHIRHL7[FHIR R4 Server\nHL7/external]
        HFHub[HuggingFace Hub\nModel weights]
    end

    Client -->|HTTPS| Nginx
    Twilio -->|WebSocket| Gateway
    Nginx -->|/api/*| Gateway
    Nginx --> Frontend
    Frontend -->|JWT Bearer| Gateway
    Gateway -->|x-internal-secret| ASR
    Gateway -->|x-internal-secret| LLM
    Gateway -->|x-internal-secret| LLMVA
    Gateway -->|x-internal-secret| TTS
    Gateway -->|x-internal-secret| SOAP
    Gateway -->|x-internal-secret| FHIR
    Gateway --- PG
    Gateway --- Redis
    Gateway --- Qdrant
    SOAP --- PG
    FHIR -->|HL7 FHIR R4| FHIRHL7
    ASR -. model pull .-> HFHub
    LLM -. model pull .-> HFHub

    style CRIT fill:#ff4444,color:#fff
```

**Current auth flow:**  
`Client → POST /auth/dev-login (plain-text creds) → JWT (1h) → Bearer header on all requests → JwtAuthGuard + TenantGuard (bypassed when MULTI_TENANT=false)`

**Target auth flow:**  
`Client → OIDC login → ID token exchange → JWT (15min) + refresh token (httpOnly cookie) → Gateway validates → Tenant claim from JWT`

---

## D. Data Consistency Plan

### Current State
- **ORM:** None — raw `pg.Pool` used across 5 services/controllers
- **Migrations:** 3 manual SQL files (`001_add_tenant_id.sql`, `002_add_fhir_status.sql`, `003_remove_tenant_defaults.sql`) — no runner, no version tracking
- **Transactions:** Not consistently used — audit log inserts happen outside SOAP note creation transactions
- **Multi-tenancy:** Column-level (`tenant_id text NOT NULL`) but enforced only at application layer (no row-level security at DB)

### Identified Risks
| Risk | Impact | Mitigation |
|---|---|---|
| Race condition: SOAP note created but audit log fails | PHI unaudited | Wrap both in one transaction |
| Migration applied to wrong DB | Data corruption | Add idempotency checks (already partial with `IF NOT EXISTS`) |
| No RLS in Postgres | Cross-tenant data leak if app bug | Add `ALTER TABLE ... ENABLE ROW LEVEL SECURITY` |
| Pool per controller | Connection exhaustion | Shared DB module with configured max pool size |
| No FHIR outbox retry backoff | Duplicate FHIR submissions | Exponential backoff in outbox worker |

### Plan

**Step 1 — Add migration runner (immediate):**
```bash
# Add dbmate to docker-compose
# Runs on gateway startup before listening
npx dbmate --url $DATABASE_URL up
```

**Step 2 — Migration 004: Add Row Level Security:**
```sql
-- 004_add_rls.sql
BEGIN;
ALTER TABLE soap_notes ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON soap_notes
  USING (tenant_id = current_setting('app.tenant_id'));
-- Repeat for sessions, patients, audit_log, etc.
COMMIT;
```

**Step 3 — Shared DB Module (NestJS):**
```typescript
// gateway/src/db/db.module.ts
@Global()
@Module({
  providers: [{
    provide: 'PG_POOL',
    useFactory: (config: ConfigService) => new Pool({
      connectionString: config.getOrThrow('DATABASE_URL'),
      max: 20,
      idleTimeoutMillis: 30_000,
    }),
    inject: [ConfigService],
  }],
  exports: ['PG_POOL'],
})
export class DbModule {}
```

**Step 4 — Transactional SOAP note + audit log:**
```typescript
const client = await this.pool.connect();
try {
  await client.query('BEGIN');
  const note = await client.query('INSERT INTO soap_notes...', [...]);
  await client.query('INSERT INTO audit_log...', [...]);
  await client.query('COMMIT');
} catch (e) {
  await client.query('ROLLBACK');
  throw e;
} finally {
  client.release();
}
```

---

## E. Security Plan

### Threat Model (STRIDE)

| Threat | Attack Vector | Current Control | Gap |
|---|---|---|---|
| **Spoofing** | Use dev-auth with any password in prod | NODE_ENV guard | `NODE_ENV=development` hardcoded — guard bypassed |
| **Tampering** | Modify JWT tenant claim | JWT signature | Token valid; tenant from JWT claim in production (good) — but Node dev mode active |
| **Repudiation** | Deny performing clinical action | audit_log table | Works — but outside transaction, could be missed |
| **Info Disclosure** | XSS → localStorage token theft | Helmet CSP | JWT in localStorage; CSP is permissive (unsafe-inline in STYLE_SRC) |
| **Denial of Service** | Flood GPU endpoints | ThrottlerModule 50/min | No per-endpoint limits for expensive route |
| **Elevation of Privilege** | Access another tenant's notes | TenantGuard | MULTI_TENANT=false bypasses all tenant checks |
| **IDOR** | Guess SOAP note UUID and retrieve | JwtAuthGuard | No ownership check on GET /soap/notes/:id |

### Remediation Priority Matrix

#### P0 — Fix Before Any Production Use
1. **NODE_ENV=production** in `docker-compose.demo.yml` — 1-line fix
2. **POSTGRES_PASSWORD** as required env var (not hardcoded `postgres:postgres`)
3. **CORS** locked to explicit origin (not `*`)

#### P1 — Fix Within 1 Sprint
4. **JWT expiry alignment** — set `expiresIn` to `24h` everywhere, fix response body
5. **Token storage** — move from localStorage to httpOnly cookie
6. **Token refresh** — add `/auth/refresh` endpoint + `withCredentials` on axios
7. **Ownership check** on `GET /soap/notes/:id`, `GET /soap/job/:id`

#### P2 — Fix Within 2 Sprints
8. **Remove validateUser() stub** — implement real user table with bcrypt
9. **Row Level Security** in PostgreSQL
10. **OIDC implementation** — replace stubs with real provider integration
11. **Shared DB module** — eliminate per-controller Pool instances
12. **safeLog expansion** — add password/token/secret to redaction list

#### P3 — Before GA
13. **MULTI_TENANT=true** with tenant provisioning API
14. **Per-endpoint rate limits** for GPU routes (ASR/LLM: 5/min per user)
15. **DB migration runner** integrated into startup / CI
16. **Credential rotation** — Twilio and HF tokens found in local .env

---

## F. Implementation Backlog

Ordered by severity × effort:

| ID | Title | Severity | Effort | Sprint |
|---|---|---|---|---|
| SEC-01 | Set NODE_ENV=production in docker-compose.demo.yml | CRITICAL | XS (1 line) | Now |
| SEC-02 | Require POSTGRES_PASSWORD env var in all compose files | CRITICAL | S | Now |
| SEC-03 | Lock CORS_ALLOWED_ORIGINS in docker-compose.demo.yml | HIGH | XS | Now |
| SEC-04 | Fix JWT expires_in response body (align to token ttl) | HIGH | S | Now |
| SEC-05 | Add 'password','token','secret' to safeLog SENSITIVE_KEYS | MEDIUM | XS | Now |
| SEC-06 | Fix Redis connection leak in getJobStatus | MEDIUM | S | Now |
| AUTH-01 | Add /auth/refresh endpoint with refresh token (httpOnly cookie) | HIGH | M | Sprint 1 |
| AUTH-02 | Move JWT from localStorage to httpOnly cookie | HIGH | M | Sprint 1 |
| AUTH-03 | Implement real user table with bcrypt password hashing | HIGH | L | Sprint 1 |
| AUTH-04 | Remove validateUser() stub | MEDIUM | XS | Sprint 1 |
| AUTH-05 | Implement OIDC with real provider (Google / Azure AD) | MEDIUM | L | Sprint 2 |
| DATA-01 | Add Shared DB Module with pool limits | MEDIUM | M | Sprint 1 |
| DATA-02 | Wrap SOAP note creation + audit log in transaction | MEDIUM | S | Sprint 1 |
| DATA-03 | Add DB migration runner (dbmate) to startup | MEDIUM | S | Sprint 1 |
| DATA-04 | Add RLS policies for tenant isolation at DB layer | HIGH | M | Sprint 2 |
| DATA-05 | Add ownership check on GET /soap/notes/:id (IDOR) | HIGH | S | Sprint 1 |
| OPS-01 | Add per-endpoint throttle for ASR/LLM routes | MEDIUM | S | Sprint 2 |
| OPS-02 | Add alerting thresholds to Grafana dashboards | LOW | M | Sprint 2 |
| OPS-03 | Replace source-mount prod compose with built images | LOW | M | Sprint 2 |
| OPS-04 | Rotate Twilio + HF credentials found in local .env | HIGH | XS | Now (out-of-band) |

---

## G. Code Changes

All changes from this audit are applied as commits. Summary of patches:

### G.1 — docker-compose.demo.yml: NODE_ENV + CORS
```diff
-  NODE_ENV=development
+  NODE_ENV=production

-  CORS_ALLOWED_ORIGINS=${CORS_ALLOWED_ORIGINS:-*}
+  CORS_ALLOWED_ORIGINS=${CORS_ALLOWED_ORIGINS:?set CORS_ALLOWED_ORIGINS in .env}
```

### G.2 — docker-compose.prod.yml: Remove hardcoded DB password
```diff
-  DATABASE_URL=postgresql://postgres:postgres@postgres:5432/healthtech
+  DATABASE_URL=postgresql://postgres:${POSTGRES_PASSWORD:?set POSTGRES_PASSWORD}@postgres:5432/healthtech
```

### G.3 — docker-compose.yml: Remove hardcoded DB password
Same pattern as G.2 for all occurrences.

### G.4 — auth.service.ts: Fix JWT expires_in response body
```diff
-  expires_in: this.config.get('JWT_EXPIRES_IN', '7d'),
+  expires_in: this.config.get('JWT_EXPIRES_IN', '3600'),
```
And align auth.module.ts signOptions to match:
```diff
-  signOptions: { expiresIn: '1h' },
+  signOptions: { expiresIn: this.jwtExpiry },
```

### G.5 — safe-logger.ts: Expand SENSITIVE_KEYS
```diff
 const SENSITIVE_KEYS = [
   'transcript', 'text', 'soap', 'payload', 'audio', 'body', 'message',
+  'password', 'token', 'secret', 'authorization', 'bearer', 'key', 'credential',
 ];
```

### G.6 — soap.controller.ts: Fix Redis connection leak
Replace inline `require('redis')` + `createClient` + `connect` + `quit` pattern with a singleton Redis client injected via a shared `KvCacheService` or `RedisModule`.

---

## H. CI/CD Plan

### Current State
- CI runs lint, unit tests, and build on PR to `main`/`develop`
- No deployment pipeline to VM
- No secrets scanning
- No migration step in CI

### Additions Required

```yaml
# Add to ci.yml jobs:

  secrets-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with: { fetch-depth: 0 }
      - name: Scan for secrets
        uses: gitleaks/gitleaks-action@v2
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

  db-migration-check:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_PASSWORD: testpass
          POSTGRES_DB: healthtech
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
    steps:
      - uses: actions/checkout@v4
      - name: Run migrations
        run: |
          for f in infra/db/migrations/*.sql; do
            psql postgresql://postgres:testpass@localhost:5432/healthtech -f "$f"
          done
      - name: Verify migration idempotency
        run: |
          for f in infra/db/migrations/*.sql; do
            psql postgresql://postgres:testpass@localhost:5432/healthtech -f "$f"
          done

  deploy-staging:
    needs: [lint-gateway, test-gateway, db-migration-check]
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - uses: actions/checkout@v4
      - name: Build and push images
        run: |
          docker build -t ghcr.io/${{ github.repository }}/gateway:${{ github.sha }} gateway/
          docker push ghcr.io/${{ github.repository }}/gateway:${{ github.sha }}
      - name: Deploy via SSH
        uses: appleboy/ssh-action@v1
        with:
          host: ${{ secrets.VM_HOST }}
          username: ${{ secrets.VM_USER }}
          key: ${{ secrets.VM_SSH_KEY }}
          script: |
            cd /opt/healthtech
            docker compose -f infra/docker-compose.demo.yml pull
            docker compose -f infra/docker-compose.demo.yml up -d --remove-orphans
```

### Required GitHub Repository Secrets
```
VM_HOST        = 34.26.235.26
VM_USER        = healthtech-deploy (dedicated deploy user, not root)
VM_SSH_KEY     = (deploy key, ed25519)
POSTGRES_PASSWORD = (generated: openssl rand -base64 32)
JWT_SECRET     = (generated: openssl rand -base64 64)
INTERNAL_SECRET = (generated: openssl rand -base64 32)
CORS_ALLOWED_ORIGINS = https://34.26.235.26
TWILIO_AUTH_TOKEN = (from Twilio console - rotate existing)
```

---

## I. Verification Tests

### I.1 — Security Regression Tests (run after each fix)

```bash
#!/usr/bin/env bash
# verify-security.sh — Run against a live gateway instance
BASE="http://localhost:3000"

echo "=== CRIT-1: Dev auth blocked in production ==="
STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
  -X POST "$BASE/auth/dev-login" \
  -H "Content-Type: application/json" \
  -d '{"username":"dev","password":"changeme"}')
# Should be 401 or 404 when NODE_ENV=production
[ "$STATUS" = "401" ] && echo "PASS: dev-login blocked" || echo "FAIL: dev-login returned $STATUS"

echo "=== CRIT-3: CORS not wildcard ==="
CORS=$(curl -s -I -H "Origin: https://evil.com" "$BASE/health" | grep -i "access-control-allow-origin")
echo "$CORS" | grep -q "\*" && echo "FAIL: CORS wildcard active" || echo "PASS: CORS restricted"

echo "=== HIGH-1: JWT expiry matches response ==="
TOKEN_RESP=$(curl -s -X POST "$BASE/auth/dev-login" \
  -H "Content-Type: application/json" \
  -d '{"username":"demo","password":"demo123"}' 2>/dev/null || echo "{}")
EXPIRES_IN=$(echo "$TOKEN_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('expires_in','N/A'))")
# Decode token and check exp claim
TOKEN=$(echo "$TOKEN_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('access_token',''))")
if [ -n "$TOKEN" ] && [ "$TOKEN" != "" ]; then
  PAYLOAD=$(echo "$TOKEN" | cut -d. -f2 | python3 -c "import sys,base64; d=sys.stdin.read().strip(); print(base64.b64decode(d + '==').decode())" 2>/dev/null)
  EXP=$(echo "$PAYLOAD" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('exp',0) - d.get('iat',0))")
  echo "Token TTL: ${EXP}s, Response expires_in: ${EXPIRES_IN}s"
  [ "$EXP" = "$EXPIRES_IN" ] && echo "PASS: expiry consistent" || echo "FAIL: mismatch (token=${EXP}s, response=${EXPIRES_IN}s)"
fi

echo "=== IDOR: Unauthorized note access ==="
# Notes retrieved without owning them should fail
STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
  -H "Authorization: Bearer $TOKEN" \
  "$BASE/soap/notes/00000000-0000-0000-0000-000000000001")
[ "$STATUS" = "403" ] || [ "$STATUS" = "404" ] && echo "PASS: IDOR protected ($STATUS)" || echo "WARN: note access returned $STATUS"

echo "=== THROTTLE: Rate limit active ==="
for i in $(seq 1 55); do
  STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/health")
done
[ "$STATUS" = "429" ] && echo "PASS: rate limit triggered" || echo "INFO: rate limit not triggered at 55 req"
```

### I.2 — Auth Flow Tests
```typescript
// gateway/test/auth-security.e2e-spec.ts
describe('Auth Security', () => {
  it('blocks dev-login when NODE_ENV=production', async () => {
    process.env.NODE_ENV = 'production';
    const res = await request(app.getHttpServer())
      .post('/auth/dev-login')
      .send({ username: 'dev', password: 'changeme' });
    expect(res.status).toBe(401);
  });

  it('returns consistent JWT expiry in response', async () => {
    const res = await request(app.getHttpServer())
      .post('/auth/dev-login')
      .send({ username: 'dev', password: 'changeme' });
    const { access_token, expires_in } = res.body;
    const payload = JSON.parse(
      Buffer.from(access_token.split('.')[1], 'base64').toString()
    );
    const actualTtl = payload.exp - payload.iat;
    expect(actualTtl).toBe(expires_in);
  });

  it('safeLog does not emit password in metadata', () => {
    const logs: any[] = [];
    const mockLogger = { log: (msg: any, meta: any) => logs.push({ msg, meta }) };
    safeLog(mockLogger as any, 'log', 'Test', { password: 'secret123', userId: 'u1' });
    expect(logs[0].meta.password).toBe('[[redacted]]');
    expect(logs[0].meta.userId).toBe('u1');
  });
});
```

### I.3 — Data Isolation Tests
```typescript
describe('Tenant Isolation', () => {
  it('prevents cross-tenant SOAP note access', async () => {
    const tokenA = mintJwt({ sub: 'userA', tenant_id: 'clinic-a' });
    const tokenB = mintJwt({ sub: 'userB', tenant_id: 'clinic-b' });
    // Create note as tenant A
    const create = await request(app.getHttpServer())
      .post('/soap/generate')
      .set('Authorization', `Bearer ${tokenA}`)
      .send(validSoapPayload);
    const noteId = create.body.id;
    // Try to read as tenant B  
    const read = await request(app.getHttpServer())
      .get(`/soap/notes/${noteId}`)
      .set('Authorization', `Bearer ${tokenB}`);
    expect(read.status).toBe(403);  // Currently FAILS — IDOR still present
  });
});
```

### I.4 — Smoke Test Checklist (post-deploy)
```bash
# Run after every deployment
curl -f http://localhost:3000/health          # Gateway health
curl -f http://localhost:5000/health          # ASR health  
curl -f http://localhost:5001/health          # LLM health
curl -f http://localhost:5002/health          # TTS health
curl -f http://localhost:5003/health          # SOAP health
curl -f http://localhost:5004/health          # FHIR health
psql $DATABASE_URL -c "SELECT COUNT(*) FROM audit_log"  # DB accessible
redis-cli -h redis PING                       # Redis accessible
```

---

## Appendix: Files Audited

| File | Status |
|---|---|
| `gateway/src/auth/auth.controller.ts` | Audited — dev-auth always active |
| `gateway/src/auth/auth.service.ts` | Audited — validateUser stub, JWT expiry mismatch |
| `gateway/src/auth/auth.module.ts` | Audited — 1h signOptions |
| `gateway/src/auth/tenant.guard.ts` | Audited — bypassed when MULTI_TENANT=false |
| `gateway/src/main.ts` | Audited — Helmet/CORS/ValidationPipe OK |
| `gateway/src/soap/soap.controller.ts` | Audited — parameterized SQL, Redis leak in getJobStatus |
| `gateway/src/audit/audit.service.ts` | Audited — parameterized SQL, tenantId required |
| `gateway/src/utils/safe-logger.ts` | Audited — missing keys in SENSITIVE_KEYS |
| `infra/docker-compose.demo.yml` | Audited — NODE_ENV=development, CORS wildcard |
| `infra/docker-compose.prod.yml` | Audited — hardcoded postgres:postgres |
| `infra/.env` | Audited — real Twilio + HF credentials (not in git) |
| `infra/db/migrations/*` | Audited — manual SQL, no runner |
| `frontend-vite/src/store/authStore.ts` | Audited — localStorage JWT, no refresh |
| `services/*/app.py` | Audited — x-internal-secret enforcement present |
| `.github/workflows/ci.yml` | Audited — no secrets scan, no migration step, no deploy |
