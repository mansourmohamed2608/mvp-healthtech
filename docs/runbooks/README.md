# Operational Runbooks

This directory contains operational procedures for the HealthTech platform.

## Runbook Index

| Runbook | Description |
|---------|-------------|
| [startup.md](./startup.md) | Service startup procedures |
| [shutdown.md](./shutdown.md) | Graceful shutdown procedures |
| [incident-response.md](./incident-response.md) | Incident response playbook |
| [scaling.md](./scaling.md) | Horizontal and vertical scaling |
| [database.md](./database.md) | Database operations and recovery |
| [rollback.md](./rollback.md) | Deployment rollback procedures |
| [monitoring.md](./monitoring.md) | Monitoring and alerting guide |
| [troubleshooting.md](./troubleshooting.md) | Common issues and solutions |

## Quick Reference

### Emergency Contacts

| Role | Contact |
|------|---------|
| On-Call Engineer | PagerDuty: `healthtech-oncall` |
| Platform Lead | escalation@example.com |
| Security Team | security@example.com |

### Critical URLs

| Service | URL |
|---------|-----|
| Production | https://healthtech.example.com |
| Staging | https://staging.healthtech.example.com |
| Grafana | https://grafana.healthtech.example.com |
| Prometheus | https://prometheus.healthtech.example.com |

### Health Check Endpoints

```bash
# Gateway health
curl https://healthtech.example.com/health

# Individual services (internal)
curl http://asr:8001/health
curl http://llm:8002/health
curl http://tts:8003/health
curl http://soap:8004/health
curl http://fhir:8005/health
```
