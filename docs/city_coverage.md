# City Coverage Guide

## Overview

This guide details the cities currently supported by the Energy Forecast Platform, including data coverage, model accuracy, and expansion plans.

## Supported Cities

### Tier 1 Cities
| City | Coverage Start | Data Quality | Model Accuracy |
|------|---------------|--------------|----------------|
| Mumbai | 2020-01-01 | High | 97.5% |
| Delhi | 2020-01-01 | High | 96.8% |
| Bangalore | 2020-01-01 | High | 97.2% |
| Chennai | 2020-01-01 | High | 96.5% |
| Kolkata | 2020-01-01 | High | 96.3% |
| Hyderabad | 2020-01-01 | High | 97.0% |

### Tier 2 Cities
| City | Coverage Start | Data Quality | Model Accuracy |
|------|---------------|--------------|----------------|
| Pune | 2021-01-01 | High | 95.8% |
| Ahmedabad | 2021-01-01 | High | 95.5% |
| Jaipur | 2021-01-01 | Medium | 94.2% |
| Lucknow | 2021-06-01 | Medium | 93.8% |
| Chandigarh | 2021-06-01 | Medium | 94.5% |
| Bhopal | 2021-06-01 | Medium | 93.5% |

## Data Sources

### Primary Sources
- State Electricity Boards
- Power Distribution Companies
- Weather Stations
- Industrial Zones

### Secondary Sources
- Satellite Data
- Population Census
- Economic Indicators
- Industrial Growth Data

## Coverage Details

### Data Points
- Energy Consumption (hourly)
- Weather Parameters
- Population Density
- Industrial Usage
- Commercial Usage
- Residential Usage

### Quality Metrics
- Data Completeness
- Accuracy Level
- Update Frequency
- Validation Status

## Expansion Plans

### Phase 1 (2024 Q1)
- Nagpur
- Indore
- Coimbatore
- Kochi

### Phase 2 (2024 Q2)
- Visakhapatnam
- Surat
- Vadodara
- Thiruvananthapuram

## Model Performance

### Accuracy Metrics
```python
{
    "MAPE": {
        "Tier1": "< 3%",
        "Tier2": "< 5%"
    },
    "MAE": {
        "Tier1": "< 2 MWh",
        "Tier2": "< 3 MWh"
    },
    "RMSE": {
        "Tier1": "< 2.5 MWh",
        "Tier2": "< 3.5 MWh"
    }
}
```

## Data Update Schedule

### Real-time Data
- Energy consumption (5-minute intervals)
- Weather data (hourly)
- Grid status (real-time)

### Periodic Updates
- Population data (quarterly)
- Economic indicators (monthly)
- Infrastructure changes (monthly)

## Integration Status

### API Endpoints
- `/cities` - List all supported cities
- `/cities/{city}/status` - Get city coverage status
- `/cities/{city}/metrics` - Get city-specific metrics

### Data Feeds
- Real-time consumption data
- Weather forecasts
- Special events calendar

## Related Documentation
- [Data Pipeline Guide](./data_pipeline.md)
- [Model Training Guide](./model_training_guide.md)
- [API Reference](./api_reference.md)
- [Performance Optimization](./performance_optimization.md)
