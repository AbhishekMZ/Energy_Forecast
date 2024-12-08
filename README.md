# Energy Forecast Platform 

A comprehensive machine learning system for predicting and optimizing energy consumption across Indian cities.

## Overview
Advanced energy demand forecasting and renewable energy optimization platform for Indian cities.

[![Tests](https://github.com/yourusername/energy_forecast/actions/workflows/tests.yml/badge.svg)](https://github.com/yourusername/energy_forecast/actions)
[![Coverage](https://codecov.io/gh/yourusername/energy_forecast/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/energy_forecast)
[![Documentation](https://readthedocs.org/projects/energy-forecast/badge/?version=latest)](https://energy-forecast.readthedocs.io)

## Features

### Forecasting
- Multi-horizon demand prediction
- Weather-aware modeling
- Confidence intervals
- Seasonal decomposition

### Renewable Energy
- Source optimization
- Weather impact analysis
- Cost optimization
- Production scheduling

### City Support
- Mumbai
- Delhi
- (More cities coming soon)

### Analytics
- Interactive dashboards
- Performance metrics
- Resource allocation
- Cost analysis

## Monitoring & Performance

### Metrics Collection
- **API Performance**
  - Request latency by endpoint
  - Request count and error rates
  - Endpoint-specific performance metrics

- **Model Performance**
  - Inference time tracking
  - Batch processing statistics
  - Model accuracy metrics
  - Prediction latency

- **Cache Performance**
  - Hit/miss ratio
  - Operation latency
  - Cache size monitoring
  - TTL statistics

- **Database Performance**
  - Query latency by type
  - Connection pool utilization
  - Query count and error rates

### Monitoring Dashboard
Access the Grafana dashboard at http://localhost:3000
- Username: `admin`
- Password: `admin123`

#### Available Panels
1. **API Performance**
   - Request latency by endpoint
   - Request rate tracking
   - Error rate monitoring

2. **Model Insights**
   - Inference time trends
   - Batch size distribution
   - Model throughput

3. **Cache Analytics**
   - Hit ratio gauge
   - Operation latency trends
   - Cache utilization

4. **Database Metrics**
   - Query latency trends
   - Connection pool status
   - Query throughput

### Alerting System
Automated alerts for:
- High API latency (>2s)
- Low cache hit ratio (<50%)
- High model inference time (>1s)
- Database connection pool saturation (>80%)
- Large batch sizes (95th percentile >100)

## Quick Start

### Prerequisites
- Python 3.8+
- PostgreSQL 13+
- Redis (optional)

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/energy_forecast.git
cd energy_forecast

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env with your configurations

# Run migrations
python manage.py migrate

# Start server
uvicorn energy_forecast.api.main:app --reload
```

## Documentation

### API Documentation
- [OpenAPI Specification](docs/api/openapi.yaml)
- [API Reference](docs/api/README.md)
- [Authentication Guide](docs/api/auth.md)

### User Guides
- [Getting Started](docs/guides/getting_started.md)
- [User Guide](docs/guides/user_guide.md)
- [Advanced Usage](docs/guides/advanced.md)

### Technical Documentation
- [Architecture](docs/technical/architecture.md)
- [Model Documentation](docs/models/model_documentation.md)
- [Deployment Guide](docs/deployment/deployment_guide.md)

## Testing

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit          # Unit tests
pytest -m integration   # Integration tests
pytest -m api          # API tests

# Generate coverage report
pytest --cov=energy_forecast --cov-report=html
```

## Dashboard

Access the interactive dashboard at `http://localhost:8000/dashboard`

Features:
- Real-time demand monitoring
- Resource allocation visualization
- Cost analysis
- Performance metrics

## Development

### Code Style
```bash
# Format code
black energy_forecast

# Check style
flake8 energy_forecast

# Type checking
mypy energy_forecast
```

### Pre-commit Hooks
```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Deployment

### Docker
```bash
# Build image
docker build -t energy-forecast .

# Run container
docker run -p 8000:8000 energy-forecast
```

### Kubernetes
```bash
# Deploy application
kubectl apply -f k8s/

# Scale deployment
kubectl scale deployment energy-forecast --replicas=3
```

## Performance

### Metrics
- Forecast Accuracy: MAPE < 5%
- Response Time: < 200ms (95th percentile)
- Throughput: 1000 req/s
- Availability: 99.9%

### Scalability
- Horizontal scaling
- Load balancing
- Cache optimization
- Database sharding

## Security

- JWT authentication
- API key validation
- Rate limiting
- Input validation
- HTTPS enforcement

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file

## Acknowledgments

- Weather data: OpenWeatherMap
- Base models: scikit-learn
- Optimization: Pyomo
- Visualization: Plotly

## Support

- GitHub Issues
- Email: support@energyforecast.com
- Documentation: https://energy-forecast.readthedocs.io
