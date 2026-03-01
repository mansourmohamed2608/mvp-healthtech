# Security Hardening Guide

## Overview

This document outlines security measures implemented in the HealthTech platform.

---

## ✅ Current Implementation Status

The following controls are **active in the deployed codebase**. Each entry includes the exact source file.

| Control | Status | Implementation |
|---|---|---|
| **Input validation** | ✅ Active | `gateway/src/main.ts` – `ValidationPipe({ whitelist: true, transform: true })` applied globally |
| **Input sanitization** | ✅ Active | `gateway/src/common/interceptors/input-sanitization.interceptor.ts` – blocks SQL/NoSQL injection, XSS, path traversal |
| **Rate limiting** | ✅ Active | `gateway/src/common/guards/enhanced-rate-limiter.guard.ts` + `@nestjs/throttler` v6.4 – per-endpoint and per-tenant limits |
| **Security headers** | ✅ Active | `gateway/src/main.ts` – `helmet()` with CSP, HSTS (1 yr), frameguard: deny, noSniff, xssFilter, referrer policy |
| **JWT authentication** | ✅ Active | `gateway/src/auth/` – HS256 JWT, validated on every protected route via `JwtAuthGuard` |
| **WebSocket auth** | ✅ Active | `gateway/src/adapters/twilio-ws.adapter.ts` – HMAC-SHA256 sig + 5-min replay window |
| **Internal service auth** | ✅ Active | All Python services validate `x-internal-secret` header on every non-health request |
| **Audit logging** | ✅ Active | `gateway/src/audit/audit.service.ts` – tenant-scoped, writes to `audit_log` table in Postgres; `tenantId` is required (throws if missing) |
| **Audit interceptor** | ✅ Active | `gateway/src/common/interceptors/audit-logging.interceptor.ts` – logs every API call with userId, tenantId, method, path, status, duration |
| **CORS** | ✅ Active | `gateway/src/main.ts` – restricted to `CORS_ALLOWED_ORIGINS` env var |
| **Gitleaks** | ✅ Active | `.github/workflows/ci.yml` – `gitleaks/gitleaks-action@v2` scans every push; `.gitleaks.toml` allowlists placeholder strings |
| **NPM audit** | ✅ Active | CI: `npm audit --audit-level=high` on gateway |
| **Container scan** | ✅ Active | CI: `aquasecurity/trivy-action` scans all Docker images |
| **Encryption in transit** | ✅ Active | TLS 1.2+ via Let's Encrypt cert on nginx; HSTS preload |
| **Encryption at rest** | ⚠️ Partial | GCP persistent disk is **not** encrypted with CMEK. See note below. |
| **WAF** | ❌ Pending | No WAF in front of nginx yet. Cloud Armor recommended for production. |

### Encryption at Rest – Gap and Remediation

GCP Compute Engine disks are encrypted by default with Google-managed keys (AES-256), but **Customer-Managed Encryption Keys (CMEK)** are not configured. For HIPAA production readiness:

```bash
# 1. Create a Cloud KMS key ring
gcloud kms keyrings create healthtech-keyring \
  --location=us-east1 --project=healthtech-482409

# 2. Create an encryption key
gcloud kms keys create healthtech-disk-key \
  --location=us-east1 \
  --keyring=healthtech-keyring \
  --purpose=encryption \
  --project=healthtech-482409

# 3. Create new disk with CMEK (migration step — requires VM stop)
gcloud compute disks create healthtech-demo-encrypted \
  --size=100GB --zone=us-east1-b \
  --kms-key=projects/healthtech-482409/locations/us-east1/keyRings/healthtech-keyring/cryptoKeys/healthtech-disk-key
```

PostgreSQL data-at-rest encryption: install `pgcrypto` extension and use `pgp_sym_encrypt()` for PHI columns.

---

## Security Layers

### 1. Network Security

#### TLS Configuration

All external traffic must use TLS 1.2+:

```nginx
# nginx-gateway.conf
ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
ssl_prefer_server_ciphers on;
ssl_session_cache shared:SSL:10m;
ssl_session_timeout 10m;

# HSTS
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
```

#### Firewall Rules

```bash
# Allow only necessary ports
ufw allow 443/tcp  # HTTPS
ufw allow 80/tcp   # HTTP (redirect to HTTPS)
ufw deny 5432/tcp  # Block direct PostgreSQL access
ufw deny 6379/tcp  # Block direct Redis access
```

### 2. Authentication & Authorization

#### JWT Configuration

```typescript
// Recommended JWT settings
{
  algorithm: 'RS256',  // Use asymmetric keys
  expiresIn: '15m',    // Short-lived access tokens
  issuer: 'healthtech',
  audience: 'healthtech-api',
}

// Refresh tokens
{
  expiresIn: '7d',
  rotation: true,      // Rotate on each use
}
```

#### Role-Based Access Control (RBAC)

```typescript
// roles.enum.ts
export enum Role {
  ADMIN = 'admin',
  CLINICIAN = 'clinician',
  NURSE = 'nurse',
  VIEWER = 'viewer',
}

// Endpoint permissions
'/soap/generate' - ['clinician', 'admin']
'/fhir/push'     - ['clinician', 'admin']
'/admin/*'       - ['admin']
```

### 3. Input Validation

#### Validation DTOs

All endpoints use class-validator DTOs:

```typescript
// See: gateway/src/common/dto/validation.dto.ts

// Example usage
@Post('transcribe')
async transcribe(@Body() dto: TranscribeRequestDto) {
  // dto is validated and sanitized
}
```

#### Sanitization

Input sanitization interceptor:
- Blocks SQL injection patterns
- Blocks NoSQL injection patterns
- Removes XSS payloads
- Validates path traversal attempts

```typescript
// See: gateway/src/common/interceptors/input-sanitization.interceptor.ts
```

### 4. Rate Limiting

#### Endpoint-Specific Limits

| Endpoint | Limit | Window |
|----------|-------|--------|
| `/auth/login` | 5 | 1 minute |
| `/asr/transcribe` | 10 | 1 minute |
| `/llm/infer` | 20 | 1 minute |
| `/llm/chat` | 30 | 1 minute |
| `/tts/synthesize` | 30 | 1 minute |

#### Tier-Based Multipliers

| Tier | Multiplier |
|------|------------|
| Free | 1x |
| Starter | 2x |
| Professional | 5x |
| Enterprise | 10x |

### 5. Security Headers

Helmet middleware configured:

```typescript
app.register(helmet, {
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      scriptSrc: ["'self'"],
      styleSrc: ["'self'", "'unsafe-inline'"],
      imgSrc: ["'self'", 'data:', 'https:'],
      connectSrc: ["'self'", 'wss:'],
    },
  },
  crossOriginEmbedderPolicy: true,
  crossOriginOpenerPolicy: true,
  crossOriginResourcePolicy: { policy: 'same-origin' },
  referrerPolicy: { policy: 'strict-origin-when-cross-origin' },
});
```

### 6. CORS Configuration

```typescript
app.enableCors({
  origin: [
    'https://healthtech.example.com',
    'https://staging.healthtech.example.com',
  ],
  methods: ['GET', 'POST', 'PUT', 'PATCH', 'DELETE'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-Tenant-ID'],
  credentials: true,
  maxAge: 86400,
});
```

### 7. Database Security

#### Connection Security

```bash
# PostgreSQL SSL
DATABASE_URL=postgresql://user:pass@host:5432/db?sslmode=require
```

#### Query Parameterization

Always use parameterized queries:

```python
# Good - parameterized
await conn.fetch("SELECT * FROM users WHERE id = $1", user_id)

# Bad - string interpolation
await conn.fetch(f"SELECT * FROM users WHERE id = {user_id}")  # NEVER DO THIS
```

### 8. Audit Logging

All API access is logged:

```typescript
// See: gateway/src/common/interceptors/audit-logging.interceptor.ts

// Log entry structure
{
  type: 'AUDIT',
  timestamp: '2024-01-15T10:30:00.000Z',
  requestId: 'req_abc123',
  userId: 'user_xyz',
  tenantId: 'tenant_123',
  action: 'CREATE',
  resource: 'soap',
  method: 'POST',
  path: '/soap/generate',
  statusCode: 200,
  duration: 1234,
  ip: '192.168.1.1',
  containsPatientData: true,
  resourceIds: {
    patientId: 'patient_456',
    encounterId: 'enc_789',
  },
}
```

### 9. Secret Management

See [SECRETS_MANAGEMENT.md](./SECRETS_MANAGEMENT.md) for details.

Key points:
- Never commit secrets
- Use environment variables or secret managers
- Rotate regularly
- Audit access

---

## Security Checklist

### Pre-Deployment

- [ ] All secrets removed from codebase
- [ ] TLS certificates configured
- [ ] Security headers enabled
- [ ] Rate limiting configured
- [ ] Input validation on all endpoints
- [ ] CORS properly configured
- [ ] Audit logging enabled
- [ ] Database connections encrypted

### Post-Deployment

- [ ] Penetration testing completed
- [ ] Vulnerability scan passed
- [ ] Security monitoring enabled
- [ ] Incident response plan documented
- [ ] Backup encryption verified
- [ ] Access reviews scheduled

### Regular Maintenance

- [ ] Secret rotation (monthly)
- [ ] Dependency updates (weekly)
- [ ] Security patch review (weekly)
- [ ] Access audit (monthly)
- [ ] Penetration test (quarterly)

---

## Incident Response

### Security Incident Classification

| Level | Description | Response Time |
|-------|-------------|---------------|
| Critical | Active breach, data exposure | Immediate |
| High | Vulnerability exploited | 1 hour |
| Medium | Potential vulnerability | 24 hours |
| Low | Security improvement | 1 week |

### Response Procedure

1. **Contain** - Isolate affected systems
2. **Assess** - Determine scope and impact
3. **Notify** - Alert security team and stakeholders
4. **Remediate** - Fix vulnerability
5. **Recover** - Restore normal operations
6. **Review** - Post-incident analysis

---

## Compliance

### HIPAA Requirements

- [ ] PHI encrypted at rest
- [ ] PHI encrypted in transit
- [ ] Access controls implemented
- [ ] Audit logging enabled
- [ ] Business Associate Agreements (BAA) in place
- [ ] Risk assessment completed
- [ ] Training completed

### SOC 2 Controls

- [ ] Access control (CC6.1)
- [ ] System operations (CC7.1)
- [ ] Change management (CC8.1)
- [ ] Risk mitigation (CC9.1)
