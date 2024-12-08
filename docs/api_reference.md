# Energy Forecast Platform API Documentation

## Overview

The Energy Forecast Platform API provides comprehensive endpoints for energy consumption prediction, data management, and system monitoring across Indian cities. This guide covers authentication, endpoints, request/response formats, rate limits, and best practices.

## Base URL

```
Production API: https://api.energyforecast.com/v1
Staging API: https://staging-api.energyforecast.com/v1
Development: http://localhost:8000/v1
```

## Authentication

### API Key Authentication

```http
Authorization: Bearer <api_key>
```

Example:
```bash
curl -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..." \
     https://api.energyforecast.com/v1/forecast
```

### JWT Authentication

```http
Authorization: JWT <token>
```

## Endpoints

### Forecasting

#### Get Energy Forecast

```http
GET /forecast
```

Query Parameters:
```json
{
    "city": "string (required) - City name",
    "start_date": "string (required) - ISO 8601 format",
    "end_date": "string (required) - ISO 8601 format",
    "granularity": "string (optional) - hourly|daily|weekly|monthly, default: daily",
    "model_version": "string (optional) - default: latest"
}
```

Response:
```json
{
    "status": "success",
    "data": {
        "forecasts": [
            {
                "timestamp": "2024-12-08T00:00:00+05:30",
                "consumption": 1234.56,
                "unit": "kWh",
                "confidence": 0.95
            }
        ],
        "metadata": {
            "model_version": "v2.1.0",
            "features_used": ["temperature", "humidity", "day_of_week"],
            "accuracy_metrics": {
                "mape": 3.2,
                "rmse": 45.6
            }
        }
    }
}
```

#### Batch Forecast

```http
POST /forecast/batch
```

Request Body:
```json
{
    "locations": [
        {
            "city": "Mumbai",
            "coordinates": {
                "latitude": 19.0760,
                "longitude": 72.8777
            }
        }
    ],
    "time_range": {
        "start": "2024-12-08T00:00:00+05:30",
        "end": "2024-12-15T23:59:59+05:30"
    },
    "options": {
        "granularity": "hourly",
        "include_weather": true,
        "model_version": "v2.1.0"
    }
}
```

### Data Management

#### Upload Training Data

```http
POST /data/upload
Content-Type: multipart/form-data
```

Form Parameters:
```
file: CSV file (required)
metadata: JSON string (optional)
```

CSV Format:
```csv
timestamp,consumption,temperature,humidity,day_type
2024-12-08T00:00:00+05:30,1234.56,25.6,65,weekday
```

Response:
```json
{
    "status": "success",
    "data": {
        "upload_id": "upload_12345",
        "rows_processed": 1000,
        "validation_errors": [],
        "storage_location": "s3://energy-forecast/training/2024/12/08/batch_001.csv"
    }
}
```

#### Query Historical Data

```http
GET /data/query
```

Query Parameters:
```json
{
    "start_date": "string (required) - ISO 8601",
    "end_date": "string (required) - ISO 8601",
    "cities": "array (optional) - List of cities",
    "metrics": "array (optional) - List of metrics to include",
    "aggregation": "string (optional) - none|hourly|daily|weekly|monthly"
}
```

### Model Management

#### List Models

```http
GET /models
```

Response:
```json
{
    "status": "success",
    "data": {
        "models": [
            {
                "version": "v2.1.0",
                "created_at": "2024-12-08T10:00:00+05:30",
                "status": "active",
                "metrics": {
                    "mape": 3.2,
                    "rmse": 45.6
                },
                "features": ["temperature", "humidity", "day_of_week"],
                "supported_cities": ["Mumbai", "Delhi", "Bangalore"]
            }
        ]
    }
}
```

#### Trigger Model Training

```http
POST /models/train
```

Request Body:
```json
{
    "training_data": {
        "start_date": "2024-01-01T00:00:00+05:30",
        "end_date": "2024-12-08T00:00:00+05:30"
    },
    "model_config": {
        "features": ["temperature", "humidity", "day_of_week"],
        "hyperparameters": {
            "learning_rate": 0.01,
            "max_depth": 6,
            "n_estimators": 100
        }
    },
    "validation_split": 0.2
}
```

### System Health

#### Health Check

```http
GET /health
```

Response:
```json
{
    "status": "healthy",
    "version": "2.1.0",
    "timestamp": "2024-12-08T22:06:00+05:30",
    "components": {
        "database": "healthy",
        "cache": "healthy",
        "model_service": "healthy"
    },
    "metrics": {
        "uptime": "5d 12h 34m",
        "request_rate": 156.7,
        "error_rate": 0.01
    }
}
```

#### System Metrics

```http
GET /metrics
```

Response:
```json
{
    "status": "success",
    "data": {
        "system": {
            "cpu_usage": 45.6,
            "memory_usage": 78.9,
            "disk_usage": 56.7
        },
        "application": {
            "active_connections": 234,
            "requests_per_second": 156.7,
            "average_response_time": 45.6
        },
        "model": {
            "inference_time_p95": 123.4,
            "cache_hit_rate": 0.89,
            "active_model_version": "v2.1.0"
        }
    }
}
```

## Error Handling

### Error Codes

| Code | Description |
|------|-------------|
| 400  | Bad Request - Invalid parameters |
| 401  | Unauthorized - Invalid or missing authentication |
| 403  | Forbidden - Insufficient permissions |
| 404  | Not Found - Resource doesn't exist |
| 429  | Too Many Requests - Rate limit exceeded |
| 500  | Internal Server Error |

### Error Response Format

```json
{
    "status": "error",
    "error": {
        "code": "FORECAST_001",
        "message": "Invalid date range specified",
        "details": {
            "start_date": "must be before end_date",
            "max_range": "1 year"
        }
    }
}
```

## Rate Limits

```
Standard Tier:
- 100 requests per minute
- 1000 requests per hour
- 10000 requests per day

Premium Tier:
- 1000 requests per minute
- 10000 requests per hour
- 100000 requests per day
```

## Webhooks

### Configuration

```http
POST /webhooks/configure
```

Request Body:
```json
{
    "url": "https://your-domain.com/webhook",
    "events": ["forecast_complete", "model_trained", "data_processed"],
    "secret": "your_webhook_secret"
}
```

### Event Types

1. Forecast Complete
```json
{
    "event": "forecast_complete",
    "timestamp": "2024-12-08T22:06:00+05:30",
    "data": {
        "request_id": "req_12345",
        "status": "success",
        "processing_time": 1.23
    }
}
```

2. Model Trained
```json
{
    "event": "model_trained",
    "timestamp": "2024-12-08T22:06:00+05:30",
    "data": {
        "model_version": "v2.1.0",
        "training_duration": 3600,
        "metrics": {
            "mape": 3.2,
            "rmse": 45.6
        }
    }
}
```

## SDK Examples

### Python

```python
from energy_forecast import Client

client = Client(api_key="your_api_key")

# Get forecast
forecast = client.get_forecast(
    city="Mumbai",
    start_date="2024-12-08T00:00:00+05:30",
    end_date="2024-12-15T23:59:59+05:30",
    granularity="hourly"
)

# Batch forecast
batch_forecast = client.batch_forecast(
    locations=[{"city": "Mumbai"}, {"city": "Delhi"}],
    start_date="2024-12-08T00:00:00+05:30",
    end_date="2024-12-15T23:59:59+05:30"
)

# Upload data
client.upload_data(
    file_path="data.csv",
    metadata={"source": "weather_station_1"}
)
```

### JavaScript

```javascript
const EnergyForecast = require('energy-forecast');

const client = new EnergyForecast({
    apiKey: 'your_api_key'
});

// Get forecast
client.getForecast({
    city: 'Mumbai',
    startDate: '2024-12-08T00:00:00+05:30',
    endDate: '2024-12-15T23:59:59+05:30',
    granularity: 'hourly'
})
.then(forecast => console.log(forecast))
.catch(error => console.error(error));

// Batch forecast
client.batchForecast({
    locations: [
        { city: 'Mumbai' },
        { city: 'Delhi' }
    ],
    timeRange: {
        start: '2024-12-08T00:00:00+05:30',
        end: '2024-12-15T23:59:59+05:30'
    }
})
.then(results => console.log(results))
.catch(error => console.error(error));
```

## Best Practices

1. **Rate Limiting**
   - Implement exponential backoff
   - Cache frequently accessed data
   - Use batch endpoints for multiple requests

2. **Error Handling**
   - Always check error responses
   - Implement proper retry logic
   - Log all API errors for debugging

3. **Security**
   - Store API keys securely
   - Use HTTPS for all requests
   - Validate webhook signatures

4. **Performance**
   - Use appropriate granularity
   - Implement response caching
   - Monitor API usage

## Support

- Email: support@energyforecast.com
- Documentation: https://docs.energyforecast.com
- Status Page: https://status.energyforecast.com
