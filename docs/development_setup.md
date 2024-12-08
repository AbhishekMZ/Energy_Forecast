# Development Environment Setup Guide

## Overview

This guide provides step-by-step instructions for setting up a development environment for the Energy Forecast Platform.

## Prerequisites

### Required Software

- Python 3.9+
- Docker Desktop
- Git
- VS Code or PyCharm
- PostgreSQL 13+
- Redis 6+

### System Requirements

- CPU: 4+ cores
- RAM: 16GB+
- Storage: 50GB+
- OS: Windows 10/11, macOS, or Linux

## Installation Steps

### 1. Python Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
.\venv\Scripts\activate
# Unix/macOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 2. Database Setup

```bash
# Start PostgreSQL container
docker run -d \
    --name energy-forecast-db \
    -e POSTGRES_USER=dev \
    -e POSTGRES_PASSWORD=dev_password \
    -e POSTGRES_DB=energy_forecast \
    -p 5432:5432 \
    postgres:13

# Initialize database
python scripts/init_db.py
```

### 3. Redis Setup

```bash
# Start Redis container
docker run -d \
    --name energy-forecast-redis \
    -p 6379:6379 \
    redis:6
```

### 4. Environment Configuration

```bash
# Copy example environment file
cp .env.example .env

# Edit .env file with your settings
# Example .env content:
DATABASE_URL=postgresql://dev:dev_password@localhost:5432/energy_forecast
REDIS_URL=redis://localhost:6379/0
API_KEY=your_development_api_key
DEBUG=true
```

## Project Structure

```
energy_forecast/
├── api/                 # API endpoints
├── core/               # Core business logic
├── data/               # Data processing
├── models/             # ML models
├── tests/              # Test suite
├── scripts/            # Utility scripts
├── docs/               # Documentation
└── config/             # Configuration files
```

## Development Tools

### 1. Code Quality Tools

```bash
# Install pre-commit hooks
pre-commit install

# Run linting
flake8 .

# Run type checking
mypy .

# Run code formatting
black .
isort .
```

### 2. Testing Tools

```bash
# Run unit tests
pytest tests/unit

# Run integration tests
pytest tests/integration

# Run with coverage
pytest --cov=core tests/
```

### 3. Database Tools

```bash
# Create new migration
alembic revision --autogenerate -m "description"

# Run migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1
```

## Local Development

### 1. Starting Development Server

```bash
# Start API server
uvicorn api.main:app --reload --port 8000

# Start Celery worker
celery -A core.tasks worker --loglevel=info

# Start Celery beat
celery -A core.tasks beat --loglevel=info
```

### 2. Development Endpoints

```python
# Example development endpoints
@app.get("/dev/reset-db")
async def reset_db():
    """Reset database to initial state."""
    await db.reset()
    return {"status": "success"}

@app.get("/dev/generate-test-data")
async def generate_test_data():
    """Generate test data."""
    await TestDataGenerator().generate()
    return {"status": "success"}
```

## Docker Development

### 1. Building Images

```bash
# Build development image
docker build -t energy-forecast-dev -f Dockerfile.dev .

# Build production image
docker build -t energy-forecast-prod .
```

### 2. Running Containers

```bash
# Start development environment
docker-compose -f docker-compose.dev.yml up -d

# View logs
docker-compose -f docker-compose.dev.yml logs -f

# Stop environment
docker-compose -f docker-compose.dev.yml down
```

## Debugging

### 1. VS Code Configuration

```json
// .vscode/launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: FastAPI",
            "type": "python",
            "request": "launch",
            "module": "uvicorn",
            "args": [
                "api.main:app",
                "--reload",
                "--port",
                "8000"
            ],
            "jinja": true,
            "justMyCode": true
        }
    ]
}
```

### 2. PyCharm Configuration

```python
# Add to api/main.py for PyCharm debugging
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Model Development

### 1. Training Environment

```bash
# Create training environment
conda create -n energy-forecast-train python=3.9
conda activate energy-forecast-train

# Install training dependencies
pip install -r requirements-train.txt
```

### 2. Model Training

```python
# Example training script
from core.models import ModelTrainer

trainer = ModelTrainer(
    model_type="lstm",
    hyperparameters={
        "layers": 2,
        "units": 64,
        "dropout": 0.2
    }
)

trainer.train(
    train_data=train_df,
    val_data=val_df,
    epochs=100
)
```

## Monitoring Setup

### 1. Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'energy-forecast'
    static_configs:
      - targets: ['localhost:8000']
```

### 2. Grafana Setup

```bash
# Start Grafana container
docker run -d \
    --name energy-forecast-grafana \
    -p 3000:3000 \
    grafana/grafana
```

## CI/CD Setup

### 1. GitHub Actions

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      - name: Run tests
        run: pytest
```

### 2. Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 21.5b2
    hooks:
      - id: black
  - repo: https://github.com/pycqa/isort
    rev: 5.8.0
    hooks:
      - id: isort
  - repo: https://github.com/pycqa/flake8
    rev: 3.9.2
    hooks:
      - id: flake8
```

## Security Configuration

### 1. Development Security Settings

```python
# config/security.py
SECURITY_CONFIG = {
    'development': {
        'allow_cors': True,
        'cors_origins': ['http://localhost:3000'],
        'require_auth': False,
        'rate_limit': {
            'enabled': False
        }
    }
}
```

### 2. SSL Setup

```bash
# Generate self-signed certificate for development
openssl req -x509 -newkey rsa:4096 -nodes \
    -out cert.pem \
    -keyout key.pem \
    -days 365
```

## Troubleshooting

### Common Issues

1. Database Connection
```bash
# Check database status
docker ps | grep postgres
docker logs energy-forecast-db
```

2. Redis Connection
```bash
# Test Redis connection
redis-cli ping
```

3. Environment Issues
```bash
# Check environment variables
python -c "import os; print(os.environ.get('DATABASE_URL'))"
```

## Additional Resources

- [API Documentation](./api_reference.md)
- [Model Architecture](./model_architecture.md)
- [Deployment Guide](./deployment_guide.md)
- [Security Guide](./security_guide.md)
- [Testing Guide](./testing_strategy.md)
