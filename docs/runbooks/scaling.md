# Scaling Runbook

## Overview

This runbook covers horizontal and vertical scaling procedures for the HealthTech platform.

---

## Scaling Indicators

### When to Scale Up

| Metric | Threshold | Action |
|--------|-----------|--------|
| CPU Usage | > 80% for 5 min | Scale horizontally |
| Memory Usage | > 85% | Scale vertically or horizontally |
| Request Latency (p99) | > 2s | Scale horizontally |
| Error Rate | > 1% | Investigate, then scale |
| Queue Depth | > 100 | Scale workers |

### When to Scale Down

| Metric | Threshold | Action |
|--------|-----------|--------|
| CPU Usage | < 30% for 30 min | Scale down |
| Memory Usage | < 40% | Consider smaller instances |
| Request Rate | < 10% of capacity | Remove replicas |

---

## Horizontal Scaling

### Docker Compose

```bash
# Scale gateway to 3 replicas
docker-compose up -d --scale gateway=3

# Scale multiple services
docker-compose up -d --scale gateway=3 --scale asr=2

# Check running replicas
docker-compose ps

# Scale back down
docker-compose up -d --scale gateway=1
```

### Kubernetes

```bash
# Scale deployment
kubectl scale deployment gateway --replicas=3

# Or use HPA (Horizontal Pod Autoscaler)
kubectl autoscale deployment gateway \
  --min=2 --max=10 --cpu-percent=70

# Check HPA status
kubectl get hpa

# View current pods
kubectl get pods -l app=gateway
```

### Load Balancer Configuration

When scaling horizontally, ensure load balancer is configured:

```nginx
# nginx.conf
upstream gateway {
    least_conn;
    server gateway1:3000;
    server gateway2:3000;
    server gateway3:3000;
}

server {
    location / {
        proxy_pass http://gateway;
    }
}
```

---

## Vertical Scaling

### Docker Compose

```yaml
# docker-compose.yml
services:
  gateway:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G
```

```bash
# Apply changes
docker-compose up -d gateway
```

### Kubernetes

```yaml
# deployment.yaml
resources:
  requests:
    cpu: "2"
    memory: "4Gi"
  limits:
    cpu: "4"
    memory: "8Gi"
```

```bash
# Apply changes
kubectl apply -f deployment.yaml
```

---

## Service-Specific Scaling

### Gateway

**Horizontal scaling** recommended:
- Stateless service
- Can scale to many replicas
- Use sticky sessions if needed for WebSocket connections

```bash
# Scale gateway
docker-compose up -d --scale gateway=5

# Configure sticky sessions in nginx
upstream gateway {
    ip_hash;  # Sticky sessions by IP
    server gateway1:3000;
    server gateway2:3000;
}
```

### ASR Service

**Vertical scaling** recommended for GPU:
- Requires GPU for optimal performance
- Scale vertically with more GPU memory
- For horizontal, need GPU per instance

```bash
# Increase GPU memory allocation
# Edit docker-compose.yml:
#   deploy:
#     resources:
#       reservations:
#         devices:
#           - driver: nvidia
#             count: 1
#             capabilities: [gpu]

# For multiple GPUs, assign specific GPU
# CUDA_VISIBLE_DEVICES=0 for first GPU
# CUDA_VISIBLE_DEVICES=1 for second GPU
```

### LLM Service

**Vertical scaling** required:
- Heavy GPU/memory requirements
- Model loading takes time
- Consider smaller models for horizontal scaling

```bash
# Scale vertically (more GPU memory)
# Use larger GPU instance

# Or use model quantization for smaller footprint
# Edit services/llm/.env:
# MODEL_QUANTIZATION=int8
# Then restart service
```

### TTS Service

**Horizontal scaling** supported:
- Can run multiple instances
- Each instance loads voice models

```bash
docker-compose up -d --scale tts=3
```

### SOAP/FHIR Services

**Horizontal scaling** supported:
- Stateless services
- Scale based on request volume

```bash
docker-compose up -d --scale soap=3 --scale fhir=3
```

---

## Database Scaling

### PostgreSQL Connection Pooling

Use PgBouncer for connection pooling:

```bash
# Add PgBouncer to docker-compose.yml
pgbouncer:
  image: edoburu/pgbouncer:latest
  environment:
    DATABASE_URL: postgres://healthtech:password@postgres:5432/healthtech
    POOL_MODE: transaction
    MAX_CLIENT_CONN: 1000
    DEFAULT_POOL_SIZE: 50
```

### PostgreSQL Read Replicas

For read-heavy workloads:

```yaml
# docker-compose.yml
postgres-replica:
  image: postgres:16
  command: |
    postgres -c wal_level=replica -c max_wal_senders=3
```

Configure gateway to use read replica for SELECT queries.

### Redis Scaling

```bash
# Redis Cluster for horizontal scaling
docker-compose up -d redis-cluster

# Or use Redis Sentinel for HA
docker-compose up -d redis-sentinel
```

---

## Auto-Scaling Configuration

### Kubernetes HPA

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: gateway-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: gateway
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Percent
          value: 10
          periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
        - type: Percent
          value: 100
          periodSeconds: 15
```

### AWS Auto Scaling

```bash
# Create Auto Scaling Group
aws autoscaling create-auto-scaling-group \
  --auto-scaling-group-name healthtech-gateway \
  --min-size 2 \
  --max-size 10 \
  --desired-capacity 2 \
  --target-group-arns arn:aws:elasticloadbalancing:...

# Create scaling policy
aws autoscaling put-scaling-policy \
  --auto-scaling-group-name healthtech-gateway \
  --policy-name scale-up \
  --policy-type TargetTrackingScaling \
  --target-tracking-configuration file://scaling-config.json
```

---

## Capacity Planning

### Baseline Metrics

Document current capacity:

```markdown
| Service | Instances | CPU | Memory | RPS Capacity |
|---------|-----------|-----|--------|--------------|
| Gateway | 2 | 2 cores | 4GB | 1000 rps |
| ASR | 1 | 4 cores + GPU | 16GB | 10 concurrent |
| LLM | 1 | 8 cores + GPU | 32GB | 5 concurrent |
| TTS | 2 | 2 cores | 4GB | 50 rps |
```

### Load Testing

```bash
# Install k6
brew install k6

# Run load test
k6 run --vus 100 --duration 5m scripts/load-test.js

# Gradually increase load
k6 run --vus 10 --duration 1m --iterations 100 scripts/load-test.js
k6 run --vus 50 --duration 2m scripts/load-test.js
k6 run --vus 100 --duration 5m scripts/load-test.js
```

---

## Scaling Checklist

Before scaling:
- [ ] Review current metrics and identify bottleneck
- [ ] Check resource availability (CPU, memory, GPU)
- [ ] Verify load balancer configuration
- [ ] Confirm database connection pool limits
- [ ] Test scaling in staging first

During scaling:
- [ ] Monitor deployment progress
- [ ] Watch for errors in logs
- [ ] Verify health checks pass
- [ ] Check load distribution

After scaling:
- [ ] Verify improved metrics
- [ ] Update capacity documentation
- [ ] Set up alerts for new thresholds
- [ ] Schedule review to scale down if needed
