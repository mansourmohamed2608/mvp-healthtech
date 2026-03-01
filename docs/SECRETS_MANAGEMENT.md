# Secrets Management Guide

## Overview

This document outlines best practices for managing secrets in the HealthTech platform.

---

## Principles

1. **Never commit secrets** - All secrets must be excluded from version control
2. **Use secret managers** - Production secrets must come from a secrets manager
3. **Rotate regularly** - Secrets should be rotated on a schedule
4. **Least privilege** - Grant minimal access to secrets
5. **Audit access** - Log all secret access

---

## Secret Categories

| Category | Examples | Rotation Period |
|----------|----------|-----------------|
| **Authentication** | JWT_SECRET, session keys | 90 days |
| **API Keys** | TWILIO_API_KEY, HUGGINGFACE_TOKEN | 180 days |
| **Database** | DATABASE_URL, connection strings | 90 days |
| **Service-to-Service** | INTERNAL_SECRET, WS_SHARED_SECRET | 30 days |
| **External Services** | FHIR credentials, OAuth secrets | Per provider policy |

---

## Secret Managers

### Recommended Options

| Provider | Service | Best For |
|----------|---------|----------|
| AWS | Secrets Manager, SSM Parameter Store | AWS deployments |
| Azure | Key Vault | Azure deployments |
| GCP | Secret Manager | GCP deployments |
| HashiCorp | Vault | Multi-cloud, on-prem |
| Doppler | Doppler | Developer-friendly, any cloud |

---

## Local Development

### Setup

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Generate local secrets:
   ```bash
   ./scripts/generate-local-secrets.sh
   ```

3. Never commit `.env` - it's in `.gitignore`

### Best Practices

- Use weak/dummy secrets for local development
- Don't use production secrets locally
- Each developer should have their own `.env`

---

## CI/CD Secrets

### GitHub Actions

Store secrets in GitHub repository settings:

```yaml
# .github/workflows/ci.yml
env:
  JWT_SECRET: ${{ secrets.JWT_SECRET }}
  DATABASE_URL: ${{ secrets.DATABASE_URL }}
```

Configure at: Repository > Settings > Secrets and variables > Actions

### Required CI Secrets

| Secret | Description |
|--------|-------------|
| `JWT_SECRET` | For running integration tests |
| `INTERNAL_SECRET` | Service-to-service auth |
| `DATABASE_URL` | Test database connection |
| `GHCR_TOKEN` | Container registry access |

---

## Production Secrets

### AWS Secrets Manager

```bash
# Create secret
aws secretsmanager create-secret \
  --name healthtech/production/jwt-secret \
  --secret-string "your-secret-value"

# Retrieve in application
aws secretsmanager get-secret-value \
  --secret-id healthtech/production/jwt-secret
```

### Azure Key Vault

```bash
# Create key vault
az keyvault create \
  --name healthtech-prod-kv \
  --resource-group healthtech-rg

# Set secret
az keyvault secret set \
  --vault-name healthtech-prod-kv \
  --name jwt-secret \
  --value "your-secret-value"

# Retrieve secret
az keyvault secret show \
  --vault-name healthtech-prod-kv \
  --name jwt-secret
```

### HashiCorp Vault

```bash
# Enable KV secrets engine
vault secrets enable -path=healthtech kv-v2

# Store secret
vault kv put healthtech/production/jwt-secret value="your-secret-value"

# Retrieve secret
vault kv get healthtech/production/jwt-secret
```

---

## Application Integration

### NestJS Gateway (Node.js)

```typescript
// config/secrets.service.ts
import { Injectable } from '@nestjs/common';
import { SecretsManager } from '@aws-sdk/client-secrets-manager';

@Injectable()
export class SecretsService {
  private client = new SecretsManager({ region: 'us-east-1' });

  async getSecret(secretId: string): Promise<string> {
    const response = await this.client.getSecretValue({ SecretId: secretId });
    return response.SecretString;
  }
}
```

### Python Services

```python
# services/common/secrets.py
import boto3
import json

def get_secret(secret_name: str) -> dict:
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response['SecretString'])
```

---

## Secret Rotation

### Automated Rotation Script

```bash
#!/bin/bash
# scripts/rotate-secrets.sh

# Rotate JWT secret
NEW_JWT_SECRET=$(openssl rand -hex 32)

# Update in secrets manager
aws secretsmanager put-secret-value \
  --secret-id healthtech/production/jwt-secret \
  --secret-string "$NEW_JWT_SECRET"

# Rolling restart to pick up new secret
kubectl rollout restart deployment/gateway

# Verify deployment
kubectl rollout status deployment/gateway
```

### Rotation Schedule

| Secret | Schedule | Owner |
|--------|----------|-------|
| JWT_SECRET | Every 90 days | Platform Team |
| DATABASE passwords | Every 90 days | DBA |
| API Keys | Every 180 days | DevOps |
| INTERNAL_SECRET | Every 30 days | Automated |

---

## Audit and Compliance

### Logging Secret Access

```typescript
// All secret access should be logged
logger.info({
  action: 'secret_access',
  secretId: secretName,
  userId: user.id,
  timestamp: new Date().toISOString(),
});
```

### Access Reviews

- Monthly: Review who has access to secrets
- Quarterly: Audit secret usage logs
- Annually: Full secrets management review

---

## Emergency Procedures

### Secret Compromise Response

1. **Immediately rotate** the compromised secret
2. **Revoke** any sessions using the old secret
3. **Audit** access logs for unauthorized use
4. **Notify** security team
5. **Document** incident

```bash
# Emergency secret rotation
./scripts/emergency-rotate.sh jwt-secret
```

### Secret Recovery

If a secret is lost:

1. Check secrets manager backup
2. Generate new secret if needed
3. Update all dependent services
4. Verify service functionality

---

## Checklist: New Environment Setup

- [ ] Create secrets manager instance
- [ ] Generate all required secrets
- [ ] Store secrets in manager
- [ ] Configure application to read from manager
- [ ] Verify secret access works
- [ ] Set up rotation schedule
- [ ] Configure access logging
- [ ] Document secret locations
