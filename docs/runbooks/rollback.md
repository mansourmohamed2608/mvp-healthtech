# Rollback Runbook

## Overview

This runbook covers deployment rollback procedures for the HealthTech platform.

---

## When to Rollback

Trigger a rollback when:
- Critical bugs introduced by new deployment
- Service health checks failing
- Error rate exceeds 5%
- P99 latency exceeds 10 seconds
- Security vulnerability discovered

---

## Pre-Rollback Checklist

- [ ] Confirm issue is deployment-related (not external)
- [ ] Identify the problematic version
- [ ] Identify the stable version to rollback to
- [ ] Notify team in incident channel
- [ ] Get approval for production rollback

---

## Quick Rollback Commands

### Docker Compose

```bash
# Rollback single service
IMAGE_TAG=v1.2.3 docker-compose up -d gateway

# Rollback all services
IMAGE_TAG=v1.2.3 docker-compose up -d

# Using specific image
docker-compose pull
docker-compose up -d --force-recreate
```

### Kubernetes

```bash
# View rollout history
kubectl rollout history deployment/gateway

# Rollback to previous version
kubectl rollout undo deployment/gateway

# Rollback to specific revision
kubectl rollout undo deployment/gateway --to-revision=2

# Check rollback status
kubectl rollout status deployment/gateway
```

---

## Service-Specific Rollback

### Gateway

```bash
# 1. Identify last stable version
docker images | grep gateway
git tag --list | tail -10

# 2. Deploy previous version
docker-compose stop gateway
docker pull ghcr.io/org/healthtech/gateway:v1.2.3
docker-compose up -d gateway

# 3. Verify rollback
curl http://localhost:3000/health
docker-compose logs gateway --tail=50
```

### Frontend

```bash
# 1. Deploy previous frontend version
docker-compose stop frontend
docker pull ghcr.io/org/healthtech/frontend:v1.2.3
docker-compose up -d frontend

# 2. Clear CDN cache if applicable
# aws cloudfront create-invalidation --distribution-id XXX --paths "/*"

# 3. Verify rollback
curl http://localhost:5173
```

### ASR Service

```bash
# 1. Stop ASR service
docker-compose stop asr

# 2. Deploy previous version
docker pull ghcr.io/org/healthtech/asr:v1.2.3
docker-compose up -d asr

# 3. Wait for model loading
docker-compose logs -f asr | grep "Model loaded"

# 4. Verify rollback
curl http://localhost:8001/health
```

### LLM Service

```bash
# ⚠️ LLM rollback takes 3-5 minutes for model loading

# 1. Stop LLM service
docker-compose stop llm

# 2. Deploy previous version
docker pull ghcr.io/org/healthtech/llm:v1.2.3
docker-compose up -d llm

# 3. Monitor model loading
docker-compose logs -f llm | grep -E "loading|ready"

# 4. Verify rollback
curl http://localhost:8002/health
```

### Database Migration Rollback

```bash
# ⚠️ DATABASE ROLLBACK IS DANGEROUS

# 1. Identify current migration version
cd gateway
npm run migration:status

# 2. Revert last migration
npm run migration:revert

# 3. Verify rollback
npm run migration:status

# For multiple migrations:
npm run migration:revert
npm run migration:revert
# ... repeat as needed
```

---

## Full Platform Rollback

For complete platform rollback to a known stable state:

```bash
#!/bin/bash
# full-rollback.sh

STABLE_VERSION="v1.2.3"

echo "Rolling back to version $STABLE_VERSION"

# 1. Enable maintenance mode
./scripts/maintenance-mode.sh enable

# 2. Stop all application services
docker-compose stop gateway asr llm tts soap fhir frontend

# 3. Pull stable versions
export IMAGE_TAG=$STABLE_VERSION
docker-compose pull

# 4. Start services
docker-compose up -d

# 5. Wait for services to be healthy
echo "Waiting for services to start..."
sleep 30

# 6. Verify health
./scripts/health-check.sh

# 7. Disable maintenance mode
./scripts/maintenance-mode.sh disable

echo "Rollback complete"
```

---

## Blue-Green Rollback

If using blue-green deployment:

```bash
# Current: Green is active, Blue is previous version

# 1. Verify Blue is healthy
./scripts/health-check.sh blue

# 2. Switch traffic to Blue
./scripts/switch-traffic.sh blue

# 3. Verify traffic is on Blue
curl -I https://healthtech.example.com | grep X-Environment

# 4. Keep Green running for potential roll-forward
# docker-compose -f docker-compose.green.yml stop
```

---

## Canary Rollback

If using canary deployment:

```bash
# 1. Set canary weight to 0%
kubectl patch service gateway -p '{"spec":{"selector":{"version":"stable"}}}'

# 2. Scale down canary
kubectl scale deployment gateway-canary --replicas=0

# 3. Delete canary deployment
kubectl delete deployment gateway-canary

# 4. Verify all traffic on stable
kubectl get pods -l app=gateway
```

---

## Post-Rollback Actions

### Immediate

1. **Verify service health**
   ```bash
   ./scripts/health-check.sh
   ```

2. **Check error rates**
   - Monitor Grafana dashboard
   - Verify error rate returning to normal

3. **Update status page**
   - Mark incident as mitigated
   - Communicate to users

### Within 1 Hour

1. **Document what happened**
   - Record timeline
   - Capture logs and metrics

2. **Notify stakeholders**
   - Send rollback notification
   - Explain next steps

3. **Create bug ticket**
   - Document the issue
   - Assign for investigation

### Within 24 Hours

1. **Root cause analysis**
   - Review deployment changes
   - Identify what caused the issue

2. **Update test suite**
   - Add tests to catch the issue
   - Improve CI/CD checks

3. **Schedule post-mortem**
   - Invite relevant team members
   - Prepare timeline and data

---

## Rollback Prevention

### Pre-Deployment Checks

```bash
# Run before any production deployment
./scripts/pre-deploy-check.sh

# Includes:
# - All tests pass
# - Security scans pass
# - Performance baseline met
# - Staging validation complete
```

### Deployment Safeguards

1. **Staged rollout**
   - Deploy to 10% of traffic first
   - Monitor for 15 minutes
   - Proceed if metrics are healthy

2. **Automatic rollback triggers**
   ```yaml
   # Kubernetes rollback policy
   spec:
     progressDeadlineSeconds: 600
     revisionHistoryLimit: 10
   ```

3. **Feature flags**
   - Use feature flags for new features
   - Disable flag instead of rollback

---

## Emergency Contacts for Rollback

| Role | Contact |
|------|---------|
| Platform Lead | platform-lead@example.com |
| On-Call Engineer | PagerDuty |
| Database Admin | dba@example.com |
| Security Team | security@example.com (for security rollbacks) |

---

## Rollback Decision Tree

```
Issue Detected
     │
     ▼
Is it deployment-related?
     │
  ┌──┴──┐
  No    Yes
  │      │
  │      ▼
  │   Severity?
  │      │
  │   ┌──┴──┬──────┐
  │   SEV1  SEV2   SEV3+
  │    │     │       │
  │    ▼     ▼       ▼
  │  Immediate  Assess   Can wait?
  │  Rollback   Options    │
  │              │      ┌──┴──┐
  │              ▼      No    Yes
  │         Hotfix      │      │
  │         Possible?   ▼      ▼
  │              │   Rollback  Schedule
  │           ┌──┴──┐           Fix
  │           No    Yes
  │           │      │
  │           ▼      ▼
  │        Rollback Deploy
  │                Hotfix
  │
  ▼
Investigate other causes
```
