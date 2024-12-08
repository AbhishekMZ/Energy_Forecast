# Development Guide

## Project Structure

```
energy_forecast/
├── api/                    # API Layer
│   ├── routes/            # API Routes
│   ├── middleware/        # API Middleware
│   └── validators/        # Request Validators
├── core/                  # Core Business Logic
│   ├── models/           # ML Models
│   ├── data/             # Data Processing
│   └── services/         # Business Services
├── infrastructure/        # Infrastructure Code
│   ├── aws/              # AWS Resources
│   ├── kubernetes/       # K8s Manifests
│   └── terraform/        # IaC
├── tests/                # Test Suite
│   ├── unit/            # Unit Tests
│   ├── integration/     # Integration Tests
│   └── performance/     # Performance Tests
├── scripts/              # Utility Scripts
├── docs/                 # Documentation
└── deployment/           # Deployment Configs
```

## Core Components

### 1. API Layer (`/api`)

The API layer handles all external communication and request processing.

#### Routes (`/api/routes`)
```python
# routes/forecast.py
from flask import Blueprint, request
from core.services import ForecastService

forecast_bp = Blueprint('forecast', __name__)

@forecast_bp.route('/forecast', methods=['GET'])
def get_forecast():
    """Get energy consumption forecast."""
    params = ForecastRequestSchema().load(request.args)
    service = ForecastService()
    return service.get_forecast(params)
```

#### Middleware (`/api/middleware`)
```python
# middleware/auth.py
from functools import wraps
from core.security import SecurityManager

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        SecurityManager.validate_token(token)
        return f(*args, **kwargs)
    return decorated
```

#### Validators (`/api/validators`)
```python
# validators/schemas.py
from marshmallow import Schema, fields

class ForecastRequestSchema(Schema):
    city = fields.Str(required=True)
    start_date = fields.DateTime(required=True)
    end_date = fields.DateTime(required=True)
    granularity = fields.Str(default='daily')
```

### 2. Core Business Logic (`/core`)

The core module contains all business logic and ML models.

#### Models (`/core/models`)
```python
# models/forecaster.py
import tensorflow as tf

class EnergyForecaster:
    def __init__(self, config):
        self.model = self._build_model(config)
        
    def _build_model(self, config):
        """Build the forecasting model architecture."""
        return tf.keras.Sequential([
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(1)
        ])
        
    def train(self, data):
        """Train the model on historical data."""
        pass
        
    def predict(self, features):
        """Generate energy consumption predictions."""
        pass
```

#### Data Processing (`/core/data`)
```python
# data/processor.py
import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self):
        self.scalers = {}
        
    def preprocess(self, data):
        """Preprocess raw data for model input."""
        return self._scale(self._clean(data))
        
    def _clean(self, data):
        """Clean raw data."""
        return data.dropna().sort_index()
        
    def _scale(self, data):
        """Scale features."""
        return (data - data.mean()) / data.std()
```

#### Services (`/core/services`)
```python
# services/forecast_service.py
from core.models import EnergyForecaster
from core.data import DataProcessor

class ForecastService:
    def __init__(self):
        self.model = EnergyForecaster()
        self.processor = DataProcessor()
        
    def get_forecast(self, params):
        """Generate forecast based on parameters."""
        features = self.processor.preprocess(params)
        predictions = self.model.predict(features)
        return self._format_response(predictions)
```

### 3. Infrastructure (`/infrastructure`)

Infrastructure code for cloud resources and deployment.

#### AWS Resources (`/infrastructure/aws`)
```terraform
# aws/main.tf
provider "aws" {
  region = "ap-south-1"
}

module "vpc" {
  source = "./modules/vpc"
  cidr_block = "10.0.0.0/16"
}

module "ecs" {
  source = "./modules/ecs"
  vpc_id = module.vpc.vpc_id
}
```

#### Kubernetes Manifests (`/infrastructure/kubernetes`)
```yaml
# kubernetes/api-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: energy-forecast-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api
  template:
    metadata:
      labels:
        app: api
    spec:
      containers:
      - name: api
        image: energy-forecast-api:latest
        ports:
        - containerPort: 8000
```

### 4. Tests (`/tests`)

Comprehensive test suite for all components.

#### Unit Tests (`/tests/unit`)
```python
# unit/test_models.py
import pytest
from core.models import EnergyForecaster

def test_model_prediction():
    model = EnergyForecaster()
    features = generate_test_features()
    predictions = model.predict(features)
    assert predictions.shape == (24,)  # 24-hour forecast
```

#### Integration Tests (`/tests/integration`)
```python
# integration/test_api.py
def test_forecast_endpoint(client):
    response = client.get('/forecast?city=Mumbai&start_date=2024-12-08')
    assert response.status_code == 200
    assert 'forecasts' in response.json
```

#### Performance Tests (`/tests/performance`)
```python
# performance/test_load.py
from locust import HttpUser, task

class APIUser(HttpUser):
    @task
    def get_forecast(self):
        self.client.get('/forecast?city=Mumbai')
```

## Development Workflow

### 1. Setup Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Set up pre-commit hooks
pre-commit install
```

### 2. Code Development Process

1. **Feature Branch Creation**
```bash
git checkout -b feature/new-feature
```

2. **Code Implementation**
```python
# Implement feature
def new_feature():
    pass

# Add tests
def test_new_feature():
    assert new_feature() == expected_result
```

3. **Local Testing**
```bash
# Run unit tests
pytest tests/unit

# Run integration tests
pytest tests/integration

# Run performance tests
locust -f tests/performance/test_load.py
```

4. **Code Review Process**
- Create pull request
- Pass automated checks
- Peer review
- Address feedback
- Merge to main

### 3. Deployment Process

1. **Build Process**
```bash
# Build Docker image
docker build -t energy-forecast-api .

# Push to registry
docker push energy-forecast-api:latest
```

2. **Deployment Steps**
```bash
# Deploy to staging
kubectl apply -f kubernetes/staging/

# Run smoke tests
pytest tests/smoke/

# Deploy to production
kubectl apply -f kubernetes/production/
```

## Development Guidelines

### 1. Code Style

```python
# Follow PEP 8
def calculate_metrics(data: pd.DataFrame) -> Dict[str, float]:
    """Calculate forecast accuracy metrics.
    
    Args:
        data: Input data containing actual and predicted values
        
    Returns:
        Dictionary of metric names and values
    """
    return {
        'mape': calculate_mape(data),
        'rmse': calculate_rmse(data)
    }
```

### 2. Documentation Standards

```python
class ModelTrainer:
    """Train and evaluate forecasting models.
    
    Attributes:
        model: The machine learning model instance
        config: Model configuration parameters
        
    Example:
        >>> trainer = ModelTrainer(config)
        >>> trainer.train(data)
        >>> metrics = trainer.evaluate(test_data)
    """
```

### 3. Git Workflow

```bash
# Feature development
git checkout -b feature/new-feature
git add .
git commit -m "feat: add new feature"
git push origin feature/new-feature

# Bug fixes
git checkout -b fix/bug-description
git add .
git commit -m "fix: fix bug description"
git push origin fix/bug-description
```

### 4. Testing Standards

```python
# Unit test example
def test_data_processor():
    """Test data preprocessing functionality."""
    processor = DataProcessor()
    
    # Arrange
    input_data = pd.DataFrame({
        'consumption': [100, 200, None, 400]
    })
    
    # Act
    processed = processor.preprocess(input_data)
    
    # Assert
    assert processed.isna().sum() == 0
    assert len(processed) == 3
```

## Monitoring and Debugging

### 1. Logging

```python
# logging_config.py
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
```

### 2. Metrics

```python
# metrics.py
from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    'request_total',
    'Total request count',
    ['method', 'endpoint']
)

RESPONSE_TIME = Histogram(
    'response_time_seconds',
    'Response time in seconds',
    ['method', 'endpoint']
)
```

### 3. Debugging Tools

```python
# debug_tools.py
import pdb
import cProfile

def debug_prediction(features):
    """Debug prediction pipeline."""
    pdb.set_trace()
    
    # Profile performance
    profiler = cProfile.Profile()
    profiler.enable()
    
    result = model.predict(features)
    
    profiler.disable()
    profiler.print_stats()
    
    return result
```

## Security Guidelines

### 1. Authentication

```python
# security/auth.py
from jose import jwt
from datetime import datetime, timedelta

def create_access_token(data: dict):
    """Create JWT access token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
```

### 2. Input Validation

```python
# security/validation.py
from pydantic import BaseModel, validator

class ForecastRequest(BaseModel):
    city: str
    start_date: datetime
    end_date: datetime
    
    @validator('end_date')
    def end_date_must_be_after_start(cls, v, values):
        if 'start_date' in values and v <= values['start_date']:
            raise ValueError('end_date must be after start_date')
        return v
```

### 3. Rate Limiting

```python
# security/rate_limit.py
from redis import Redis
from datetime import datetime

class RateLimiter:
    def __init__(self):
        self.redis = Redis()
        
    def is_allowed(self, user_id: str) -> bool:
        """Check if request is within rate limits."""
        key = f"rate_limit:{user_id}"
        current = self.redis.get(key) or 0
        
        if int(current) >= RATE_LIMIT:
            return False
            
        self.redis.incr(key)
        self.redis.expire(key, WINDOW_SECONDS)
        return True
```

## Performance Optimization

### 1. Caching

```python
# cache/manager.py
from functools import lru_cache
from redis import Redis

class CacheManager:
    def __init__(self):
        self.redis = Redis()
        
    @lru_cache(maxsize=1000)
    def get_forecast(self, city: str, date: str):
        """Get cached forecast."""
        key = f"forecast:{city}:{date}"
        return self.redis.get(key)
        
    def set_forecast(self, city: str, date: str, data: dict):
        """Cache forecast results."""
        key = f"forecast:{city}:{date}"
        self.redis.setex(key, CACHE_TTL, data)
```

### 2. Database Optimization

```python
# database/optimization.py
from sqlalchemy import create_engine, text

def optimize_queries():
    """Optimize database queries."""
    engine = create_engine(DATABASE_URL)
    
    with engine.connect() as conn:
        # Create indexes
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_consumption_city_date 
            ON consumption (city, date)
        """))
        
        # Update statistics
        conn.execute(text("ANALYZE consumption"))
```

## Continuous Integration/Deployment

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
      - name: Run tests
        run: |
          pip install -r requirements.txt
          pytest tests/
```

### 2. Deployment Automation

```python
# scripts/deploy.py
import subprocess
import sys

def deploy_to_environment(env: str):
    """Deploy application to specified environment."""
    try:
        # Build and push Docker image
        subprocess.run(['docker', 'build', '-t', f'energy-forecast:{env}', '.'])
        subprocess.run(['docker', 'push', f'energy-forecast:{env}'])
        
        # Deploy to Kubernetes
        subprocess.run(['kubectl', 'apply', '-f', f'kubernetes/{env}/'])
        
        # Run smoke tests
        subprocess.run(['pytest', 'tests/smoke/'])
        
    except subprocess.CalledProcessError as e:
        print(f"Deployment failed: {e}")
        sys.exit(1)
```

## Troubleshooting Guide

### 1. Common Issues

```python
# troubleshoot/common.py
def diagnose_prediction_error(error):
    """Diagnose prediction pipeline errors."""
    if isinstance(error, ValueError):
        return "Invalid input data format"
    elif isinstance(error, MemoryError):
        return "Insufficient memory for model inference"
    elif isinstance(error, TimeoutError):
        return "Prediction timeout - check model performance"
    return "Unknown error"
```

### 2. Debug Tools

```python
# troubleshoot/tools.py
def analyze_model_performance(model, data):
    """Analyze model performance issues."""
    metrics = {
        'inference_time': [],
        'memory_usage': [],
        'prediction_variance': []
    }
    
    for batch in data:
        start_time = time.time()
        prediction = model.predict(batch)
        metrics['inference_time'].append(time.time() - start_time)
        metrics['prediction_variance'].append(np.var(prediction))
        
    return metrics
```

This development guide provides a comprehensive overview of the project structure, development workflow, and best practices. Each section includes practical examples and detailed explanations to help developers understand and contribute to the project effectively.
