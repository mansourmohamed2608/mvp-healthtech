# Operational Runbooks

This directory contains operational procedures for the HealthTech platform.

## Runbook Index

| Runbook | Description |
|---------|-------------|
| [startup-shutdown.md](./startup-shutdown.md) | **GCP VM** startup, shutdown, demo checklist |
| [startup.md](./startup.md) | Local dev service startup procedures |
| [shutdown.md](./shutdown.md) | Graceful shutdown procedures |
| [incident-response.md](./incident-response.md) | Incident response playbook (GCP triage first) |
| [rollback.md](./rollback.md) | Deployment rollback procedures (GCP fast rollback) |
| [scaling.md](./scaling.md) | Horizontal and vertical scaling |
| [database.md](./database.md) | Database operations and recovery |
| [monitoring.md](./monitoring.md) | Monitoring and alerting guide |

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
