# Quick Start Guide

## Prerequisites

### Required Software
- Python 3.8+
- Docker Desktop
- Git

### System Requirements
- Memory: 8GB minimum
- Storage: 20GB free space
- CPU: 4 cores recommended

## Installation Steps

1. **Clone Repository**
```bash
git clone https://github.com/yourusername/energy_forecast.git
cd energy_forecast
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Environment Setup**
```bash
# Copy example environment file
cp .env.example .env

# Edit with your settings
# Required variables:
REDIS_URL=redis://localhost:6379
MODEL_BATCH_SIZE=32
```

4. **Start Services**
```bash
# Start monitoring infrastructure
./start_monitoring.bat

# Start application
python -m energy_forecast
```

## Verification Steps

1. **Check Services**
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090
- Application: http://localhost:8000

2. **Test API**
```bash
curl http://localhost:8000/api/v1/health
```

3. **Monitor Metrics**
- Access Grafana dashboard
- Check basic metrics
- Verify data collection

## Basic Usage

### Get Energy Forecast
```python
import requests

response = requests.get(
    "http://localhost:8000/api/v1/forecast",
    params={
        "city": "Mumbai",
        "date": "2024-12-08"
    }
)
print(response.json())
```

### Batch Forecast
```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/batch_forecast",
    json={
        "locations": [
            {
                "city": "Mumbai",
                "date": "2024-12-08"
            },
            {
                "city": "Delhi",
                "date": "2024-12-08"
            }
        ]
    }
)
print(response.json())
```

## Common Issues

### 1. Service Start Failures
- Check Docker is running
- Verify port availability
- Check environment variables

### 2. API Errors
- Validate input format
- Check service health
- Review error messages

### 3. Performance Issues
- Monitor resource usage
- Check batch sizes
- Verify cache operation

## Next Steps

1. [Configuration Guide](configuration.md)
2. [API Documentation](../api/overview.md)
3. [Monitoring Setup](../deployment/monitoring_setup.md)
