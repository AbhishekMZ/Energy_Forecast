# API Documentation

## Overview

The Energy Forecast Platform API provides endpoints for predicting energy consumption across Indian cities. This RESTful API supports both single and batch predictions, with comprehensive monitoring and caching capabilities.

## Base URL
```
http://localhost:8000/api/v1
```

## Authentication

All API requests require authentication using API keys passed in the header:
```
Authorization: Bearer YOUR_API_KEY
```

## Rate Limits

- Standard Tier: 100 requests/minute
- Premium Tier: 1000 requests/minute
- Batch API: 10 requests/minute

## Endpoints

### 1. Single Forecast
```http
GET /forecast
```

**Parameters:**
- `city` (required): Name of the city
- `date` (required): Target date (YYYY-MM-DD)
- `include_weather` (optional): Include weather data

**Response:**
```json
{
    "city": "Mumbai",
    "date": "2024-12-08",
    "forecast": {
        "consumption_kwh": 1500000,
        "confidence": 0.95,
        "prediction_time": "2024-12-08T21:08:32+05:30"
    },
    "weather": {
        "temperature": 28,
        "humidity": 75
    }
}
```

### 2. Batch Forecast
```http
POST /batch_forecast
```

**Request Body:**
```json
{
    "locations": [
        {
            "city": "Mumbai",
            "date": "2024-12-08"
        },
        {
            "city": "Delhi",
            "date": "2024-12-08"
        }
    ],
    "include_weather": true
}
```

**Response:**
```json
{
    "predictions": [
        {
            "city": "Mumbai",
            "forecast": {
                "consumption_kwh": 1500000,
                "confidence": 0.95
            }
        },
        {
            "city": "Delhi",
            "forecast": {
                "consumption_kwh": 2000000,
                "confidence": 0.93
            }
        }
    ],
    "batch_id": "batch_123",
    "processing_time": 0.5
}
```

### 3. Health Check
```http
GET /health
```

**Response:**
```json
{
    "status": "healthy",
    "version": "1.0.0",
    "timestamp": "2024-12-08T21:08:32+05:30",
    "services": {
        "database": "up",
        "cache": "up",
        "model_server": "up"
    }
}
```

## Error Handling

### Error Codes
- 400: Bad Request
- 401: Unauthorized
- 403: Forbidden
- 404: Not Found
- 429: Too Many Requests
- 500: Internal Server Error

### Error Response Format
```json
{
    "error": {
        "code": "INVALID_INPUT",
        "message": "Invalid city name provided",
        "details": {
            "field": "city",
            "value": "InvalidCity"
        }
    },
    "request_id": "req_123"
}
```

## Monitoring

### Available Metrics
- Request latency
- Cache hit ratio
- Error rates
- Model inference time

### Metric Endpoints
```http
GET /metrics
```

Response includes Prometheus-formatted metrics:
```
# HELP http_requests_total Total HTTP requests
# TYPE http_requests_total counter
http_requests_total{method="get",endpoint="/forecast"} 1000

# HELP http_request_duration_seconds HTTP request duration in seconds
# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_bucket{le="0.1",method="get",endpoint="/forecast"} 800
```

## Best Practices

1. **Rate Limiting**
   - Implement exponential backoff
   - Use batch API for multiple requests
   - Cache frequently accessed data

2. **Error Handling**
   - Always check error responses
   - Implement retry logic
   - Log error details

3. **Performance**
   - Use batch API when possible
   - Implement caching
   - Monitor response times

4. **Security**
   - Rotate API keys regularly
   - Use HTTPS
   - Validate all inputs

## SDK Support

Official SDKs available for:
- Python
- JavaScript
- Java
- Go

Example (Python):
```python
from energy_forecast import Client

client = Client(api_key="YOUR_API_KEY")
forecast = client.get_forecast(
    city="Mumbai",
    date="2024-12-08"
)
print(forecast.consumption_kwh)
```
