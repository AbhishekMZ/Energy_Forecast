# Getting Started Guide

## Quick Start 🚀

### Prerequisites
- Python 3.8+
- PostgreSQL 13+
- Redis (optional)
- Git

### Installation

1. **Clone Repository**
```bash
git clone https://github.com/yourusername/energy_forecast.git
cd energy_forecast
```

2. **Create Virtual Environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Environment Setup**
```bash
# Copy example env file
cp .env.example .env

# Edit .env with your settings
SECRET_KEY=your-secret-key
API_KEY=your-api-key
DATABASE_URL=postgresql://user:pass@localhost/energy_forecast
REDIS_URL=redis://localhost:6379/0
```

5. **Database Setup**
```bash
# Create database
createdb energy_forecast

# Run migrations
python manage.py migrate
```

6. **Start Server**
```bash
uvicorn energy_forecast.api.main:app --reload
```

## Basic Usage 📊

### 1. Generate Forecast

```python
import requests

# Setup client
api_key = "your-api-key"
headers = {"X-API-Key": api_key}

# Prepare request
forecast_request = {
    "city": "Mumbai",
    "start_date": "2023-10-01T00:00:00",
    "end_date": "2023-10-07T00:00:00",
    "weather_forecast": [{
        "temperature": 30.0,
        "humidity": 60.0,
        "cloud_cover": 50.0
    }]
}

# Make request
response = requests.post(
    "http://localhost:8000/forecast/demand",
    headers=headers,
    json=forecast_request
)

# Process results
forecast = response.json()
print(f"Demand Forecast: {forecast['demand_forecast']}")
print(f"Energy Mix: {forecast['energy_mix']}")
```

### 2. Validate Data

```python
# Prepare weather data
weather_data = {
    "temperature": [25.0, 26.0, 27.0],
    "humidity": [60.0, 65.0, 70.0],
    "cloud_cover": [30.0, 40.0, 50.0]
}

# Validate data
response = requests.post(
    "http://localhost:8000/data/validate",
    headers=headers,
    params={"city": "Mumbai"},
    json=weather_data
)

# Check results
validation = response.json()
print(f"Validation Status: {validation['status']}")
print(f"Quality Metrics: {validation['quality_metrics']}")
```

### 3. List Cities

```python
# Get available cities
response = requests.get(
    "http://localhost:8000/forecast/cities",
    headers=headers
)

cities = response.json()
print(f"Available Cities: {cities}")
```

## Basic Configuration ⚙️

### 1. Model Settings

```python
# config/constants.py
MODEL_SETTINGS = {
    'forecast_horizon': 168,  # 7 days in hours
    'confidence_level': 0.95,
    'min_training_samples': 1000
}
```

### 2. City Configuration

```python
# config/constants.py
CITY_SETTINGS = {
    'Mumbai': {
        'peak_hours': [9, 10, 11, 17, 18, 19],
        'base_demand': 1000,  # MW
        'population': 20_000_000
    },
    'Delhi': {
        'peak_hours': [8, 9, 10, 18, 19, 20],
        'base_demand': 1200,  # MW
        'population': 19_000_000
    }
}
```

### 3. Validation Rules

```python
# config/constants.py
VALIDATION_RULES = {
    'temperature': {
        'min': -10,
        'max': 50
    },
    'humidity': {
        'min': 0,
        'max': 100
    },
    'cloud_cover': {
        'min': 0,
        'max': 100
    }
}
```

## Basic Customization 🛠️

### 1. Add New City

```python
# config/constants.py
CITY_SETTINGS['Bangalore'] = {
    'peak_hours': [9, 10, 11, 17, 18, 19],
    'base_demand': 800,  # MW
    'population': 12_000_000
}
```

### 2. Modify Validation Rules

```python
# config/constants.py
VALIDATION_RULES['temperature']['max'] = 45  # Stricter limit
```

### 3. Adjust Model Parameters

```python
# config/constants.py
MODEL_SETTINGS.update({
    'forecast_horizon': 24,  # 1 day
    'confidence_level': 0.90
})
```

## Common Tasks 📋

### 1. Monitor System Health

```bash
# Check API health
curl http://localhost:8000/health

# Check logs
tail -f logs/energy_forecast.log
```

### 2. Update Dependencies

```bash
# Update all packages
pip install -r requirements.txt --upgrade

# Update specific package
pip install --upgrade fastapi
```

### 3. Run Tests

```bash
# Run all tests
pytest

# Run specific tests
pytest tests/test_api/
pytest -k "test_forecast"
```

## Troubleshooting 🔍

### Common Issues

1. **Connection Error**
   ```python
   # Check database connection
   import psycopg2
   conn = psycopg2.connect(DATABASE_URL)
   ```

2. **Authentication Error**
   ```python
   # Verify API key
   headers = {"X-API-Key": "your-api-key"}
   response = requests.get("/health", headers=headers)
   ```

3. **Validation Error**
   ```python
   # Check data format
   print(weather_data.keys())
   print(weather_data['temperature'].dtype)
   ```

### Getting Help

1. **Documentation**
   - API docs: `/docs`
   - OpenAPI spec: `/openapi.json`
   - User guide: `/docs/guides/`

2. **Support**
   - GitHub Issues
   - Email: support@energyforecast.com
   - Community forums

## Next Steps 🎯

1. Explore advanced features
2. Customize models
3. Add monitoring
4. Optimize performance

For more details, see:
- [Advanced Guide](advanced.md)
- [API Documentation](../api/README.md)
- [Model Documentation](../models/model_documentation.md)
