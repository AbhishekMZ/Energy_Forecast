# API Versioning Guide

## Overview

This guide outlines the API versioning strategy for the Energy Forecast Platform, ensuring backward compatibility while enabling platform evolution.

## Versioning Strategy

### Version Format
```
v{MAJOR}.{MINOR}
Example: v1.0, v2.1
```

### Version Components
- MAJOR: Breaking changes
- MINOR: Backward-compatible changes

## API Endpoints

### Current Versions
```python
API_VERSIONS = {
    "v1": {
        "status": "stable",
        "released": "2024-01-01",
        "sunset": "2025-01-01"
    },
    "v2": {
        "status": "beta",
        "released": "2024-12-01",
        "sunset": None
    }
}
```

### URL Structure
```
https://api.energyforecast.com/v1/forecast
https://api.energyforecast.com/v2/forecast
```

## Version Management

### Header-based Versioning
```http
Accept: application/json; version=1.0
```

### URL-based Versioning
```
/api/v1/forecast
/api/v2/forecast
```

## Compatibility Guidelines

### Breaking Changes
- Removing endpoints
- Changing response structure
- Modifying required parameters

### Non-breaking Changes
- Adding optional parameters
- Extending response data
- Adding new endpoints

## Version Lifecycle

### Beta Phase
- Feature development
- Early access
- Feedback collection
- Documentation updates

### Stable Phase
- Production-ready
- Full documentation
- Support commitment
- Performance optimized

### Deprecation Phase
- Announcement period
- Migration guide
- Limited support
- Sunset timeline

### Sunset Phase
- Read-only mode
- Final shutdown
- Archive access
- Data migration

## Migration Guidelines

### Version Migration
```python
def handle_request(version: str):
    """Handle API request based on version."""
    if version == "v1":
        return v1_handler()
    elif version == "v2":
        return v2_handler()
    else:
        raise UnsupportedVersionError()
```

### Compatibility Layer
```python
def v2_to_v1_response(data: dict) -> dict:
    """Convert v2 response to v1 format."""
    return {
        "forecast": data["prediction"],
        "confidence": data["accuracy"],
        "timestamp": data["prediction_time"]
    }
```

## Documentation Standards

### Version-specific Docs
- Endpoint specifications
- Request/response formats
- Migration guides
- Example code

### Changelog Format
```markdown
## [2.0.0] - 2024-12-01
### Added
- Real-time forecasting
- Confidence intervals
- Batch predictions

### Changed
- Response format
- Authentication method
- Rate limits

### Deprecated
- Legacy endpoints
- Old response format
```

## Testing Requirements

### Version Tests
```python
def test_api_version_compatibility():
    """Test API version compatibility."""
    v1_response = client.get("/v1/forecast")
    v2_response = client.get("/v2/forecast")
    
    assert v1_response.schema == V1_SCHEMA
    assert v2_response.schema == V2_SCHEMA
```

## Monitoring and Metrics

### Version Usage
- Requests per version
- Error rates
- Response times
- Client distribution

### Alerts
```yaml
alerts:
  - name: deprecated_version_usage
    condition: v1_requests > 100
    severity: warning
    message: "High usage of deprecated v1 API"
```

## Client Libraries

### Version Support
```python
class EnergyForecastClient:
    def __init__(self, version="v2"):
        self.version = version
        self.base_url = f"https://api.energyforecast.com/{version}"
```

### Migration Tools
```python
def migrate_to_v2(client):
    """Migrate client code to v2 API."""
    if client.version == "v1":
        # Update client configuration
        client.version = "v2"
        # Transform request/response formats
        client.transform = v2_transformer
```

## Related Documentation
- [API Reference](./api_reference.md)
- [Release Process](./release_process.md)
- [Testing Guide](./testing_guide.md)
- [Monitoring Guide](./monitoring_guide.md)
