# Deployment Guide

## System Requirements

### Hardware Requirements
- CPU: 4+ cores
- RAM: 16GB minimum
- Storage: 100GB SSD
- Network: 1Gbps

### Software Requirements
- Python 3.8+
- Docker 20.10+
- Kubernetes 1.20+
- PostgreSQL 13+

## Development Environment

### Local Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest

# Start development server
uvicorn energy_forecast.api.main:app --reload
```

### Docker Setup
```bash
# Build image
docker build -t energy-forecast:latest .

# Run container
docker run -p 8000:8000 energy-forecast:latest
```

## Production Deployment

### 1. Infrastructure Setup

#### Database
```bash
# Create PostgreSQL instance
kubectl apply -f k8s/postgres.yaml

# Initialize database
kubectl exec -it postgres-0 -- psql -U postgres
CREATE DATABASE energy_forecast;
```

#### Redis Cache
```bash
# Deploy Redis
kubectl apply -f k8s/redis.yaml
```

#### Load Balancer
```bash
# Configure Nginx
kubectl apply -f k8s/nginx.yaml
```

### 2. Application Deployment

#### Deploy API
```bash
# Deploy application
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/api-service.yaml

# Scale deployment
kubectl scale deployment energy-forecast --replicas=3
```

#### Configure Autoscaling
```bash
# Set up HPA
kubectl apply -f k8s/hpa.yaml
```

### 3. Monitoring Setup

#### Prometheus
```bash
# Deploy Prometheus
kubectl apply -f k8s/prometheus/

# Configure alerts
kubectl apply -f k8s/prometheus/alerts.yaml
```

#### Grafana
```bash
# Deploy Grafana
kubectl apply -f k8s/grafana/

# Import dashboards
kubectl apply -f k8s/grafana/dashboards/
```

## Security Configuration

### 1. SSL/TLS
```bash
# Generate certificates
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout private.key -out certificate.crt

# Configure in Kubernetes
kubectl create secret tls energy-forecast-tls \
  --key private.key --cert certificate.crt
```

### 2. Network Policies
```bash
# Apply network policies
kubectl apply -f k8s/network-policies.yaml
```

### 3. RBAC
```bash
# Configure service accounts
kubectl apply -f k8s/rbac.yaml
```

## Backup and Recovery

### Database Backup
```bash
# Automated backup
kubectl apply -f k8s/backup/cronjob.yaml

# Manual backup
kubectl exec postgres-0 -- pg_dump -U postgres energy_forecast \
  > backup.sql
```

### Model Artifacts
```bash
# Backup model files
kubectl cp energy-forecast-0:/app/models ./models-backup
```

## Monitoring and Maintenance

### Health Checks
```bash
# Check pod health
kubectl get pods -l app=energy-forecast

# View logs
kubectl logs -l app=energy-forecast
```

### Performance Monitoring
```bash
# Resource usage
kubectl top pods
kubectl top nodes

# Metrics
kubectl port-forward svc/prometheus 9090:9090
```

### Scaling Guidelines
- CPU usage > 70%: Scale up
- Memory usage > 80%: Investigate leaks
- Response time > 500ms: Check bottlenecks

## Troubleshooting

### Common Issues

1. **High Memory Usage**
   - Check memory leaks
   - Adjust resource limits
   - Monitor garbage collection

2. **Slow Response Times**
   - Check database indexes
   - Monitor cache hit rate
   - Analyze slow queries

3. **Connection Issues**
   - Verify network policies
   - Check DNS resolution
   - Validate certificates

### Debug Commands
```bash
# Check logs
kubectl logs -f deployment/energy-forecast

# Debug network
kubectl exec -it energy-forecast-0 -- netstat -tulpn

# Monitor resources
kubectl describe pod energy-forecast-0
```

## Maintenance Procedures

### Updates
1. Create backup
2. Apply database migrations
3. Deploy new version
4. Verify health checks
5. Monitor metrics

### Rollback
```bash
# Rollback deployment
kubectl rollout undo deployment/energy-forecast

# Verify rollback
kubectl rollout status deployment/energy-forecast
```
