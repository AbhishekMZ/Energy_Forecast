# Energy Forecast User Guide

## Getting Started

### Installation
```bash
# Clone repository
git clone https://github.com/your-org/energy_forecast.git

# Install dependencies
pip install -r requirements.txt
```

### Configuration
1. Create `.env` file:
```env
SECRET_KEY=your-secret-key
API_KEY=your-api-key
```

2. Configure city parameters in `config/constants.py`

## Using the API

### Authentication
```python
import requests

headers = {
    "X-API-Key": "your-api-key",
    "Authorization": "Bearer your-jwt-token"
}
```

### Generate Forecast
```python
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
demand = forecast["demand_forecast"]
energy_mix = forecast["energy_mix"]
```

### Validate Data
```python
# Validate weather data
response = requests.post(
    "http://localhost:8000/data/validate",
    headers=headers,
    params={"city": "Mumbai"},
    json=weather_data
)

validation_results = response.json()
```

## Data Formats

### Weather Data
```python
{
    "temperature": float,    # Celsius
    "humidity": float,       # Percentage
    "cloud_cover": float,    # Percentage
    "solar_radiation": float,# W/m²
    "wind_speed": float,     # m/s
    "precipitation": float   # mm
}
```

### Forecast Output
```python
{
    "demand_forecast": {
        "timestamp": float  # kWh
    },
    "energy_mix": {
        "solar": float,
        "wind": float,
        "hydro": float
    },
    "confidence_intervals": {
        "lower": float,
        "upper": float
    }
}
```

## Best Practices

### Data Quality
1. Ensure complete weather data
2. Check value ranges
3. Validate timestamps
4. Remove duplicates

### API Usage
1. Handle rate limits
2. Implement error handling
3. Cache responses
4. Use batch requests

### Performance
1. Optimize request frequency
2. Monitor response times
3. Handle timeouts
4. Implement retries

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   - Check API key
   - Verify JWT token
   - Check permissions

2. **Validation Errors**
   - Verify data formats
   - Check value ranges
   - Ensure complete data

3. **Rate Limiting**
   - Implement backoff
   - Use batch requests
   - Cache responses

### Error Codes

- 400: Invalid request
- 401: Unauthorized
- 403: Forbidden
- 429: Rate limit exceeded
- 500: Server error

## Support

- GitHub Issues: [Link]
- Documentation: [Link]
- Email: support@energyforecast.com
