# Monitoring Infrastructure Setup

## Prerequisites
- Docker Desktop
- Docker Compose
- Access to ports 3000, 9090, 6379

## Quick Start

1. **Start Monitoring Services**
```bash
./start_monitoring.bat
```

2. **Access Points**
- Grafana: http://localhost:3000 (admin/admin123)
- Prometheus: http://localhost:9090
- Redis: localhost:6379

## Detailed Setup

### 1. Docker Compose Configuration

```yaml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./monitoring/prometheus:/etc/prometheus
    ports:
      - "9090:9090"
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'

  grafana:
    image: grafana/grafana:latest
    volumes:
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
      - GF_PATHS_PROVISIONING=/etc/grafana/provisioning

  redis:
    image: redis:latest
    ports:
      - "6379:6379"
```

### 2. Prometheus Configuration

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'energy_forecast'
    static_configs:
      - targets: ['localhost:8000']
```

### 3. Grafana Setup

1. **Datasource Configuration**
```yaml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
```

2. **Dashboard Provisioning**
```yaml
apiVersion: 1

providers:
  - name: 'Energy Forecast'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    editable: true
    options:
      path: /var/lib/grafana/dashboards
```

### 4. Alert Configuration

```yaml
apiVersion: 1

groups:
  - name: EnergyForecastAlerts
    folder: Energy Forecast
    interval: 1m
    rules:
      - name: High API Latency
        condition: avg() > 2
        data:
          - refId: A
            datasourceUid: prometheus
            model:
              expr: rate(request_latency_seconds_sum[5m]) / rate(request_latency_seconds_count[5m])
```

## Maintenance

### 1. Log Rotation
```bash
# logrotate configuration
/var/log/energy_forecast/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0640 energy_forecast energy_forecast
}
```

### 2. Metric Retention
- Prometheus: 15 days default
- Adjust in prometheus.yml:
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  storage.tsdb.retention.time: 15d
```

### 3. Cache Cleanup
```python
# Redis cache cleanup script
async def cleanup_expired_cache():
    """Remove expired cache entries."""
    pattern = "forecast:*"
    cursor = 0
    while True:
        cursor, keys = await redis.scan(cursor, match=pattern)
        for key in keys:
            if not await redis.ttl(key):
                await redis.delete(key)
        if cursor == 0:
            break
```

## Troubleshooting

### 1. Service Health Check
```bash
# Check service status
docker-compose ps

# View service logs
docker-compose logs -f [service_name]
```

### 2. Common Issues

1. **Prometheus Not Scraping**
- Check target status in Prometheus UI
- Verify network connectivity
- Check port accessibility

2. **Grafana Dashboard Empty**
- Verify Prometheus data source
- Check metric availability
- Review dashboard queries

3. **Redis Connection Issues**
- Check Redis connection string
- Verify network access
- Review authentication

### 3. Performance Issues

1. **High Memory Usage**
```bash
# Check container memory usage
docker stats

# Adjust container limits in docker-compose.yml
services:
  prometheus:
    deploy:
      resources:
        limits:
          memory: 2G
```

2. **High CPU Usage**
- Review batch sizes
- Check query complexity
- Monitor container resources

3. **Slow Queries**
- Review query patterns
- Check index usage
- Optimize cache usage

## Security

### 1. Network Security
- Use internal Docker network
- Restrict port access
- Enable TLS where possible

### 2. Authentication
- Change default passwords
- Use environment variables
- Implement role-based access

### 3. Data Protection
- Enable Redis persistence
- Regular backups
- Encrypt sensitive data

## Scaling

### 1. Prometheus
- Federation for large deployments
- Remote storage integration
- Load balancing

### 2. Grafana
- Multiple instances
- Shared storage
- Load balancing

### 3. Redis
- Cluster mode
- Sentinel for HA
- Persistence configuration
