# Advanced Usage Guide

## Advanced Features

### 1. Custom Model Integration

#### Create Custom Model
```python
from energy_forecast.core.models import BaseModel

class CustomModel(BaseModel):
    def __init__(self, params):
        super().__init__()
        self.params = params
    
    def train(self, X, y):
        # Custom training logic
        pass
    
    def predict(self, X):
        # Custom prediction logic
        pass
```

#### Register Model
```python
from energy_forecast.core.registry import ModelRegistry

registry = ModelRegistry()
registry.register("custom_model", CustomModel)
```

### 2. Feature Engineering

#### Custom Features
```python
def create_weather_features(df):
    """Create advanced weather features"""
    # Temperature changes
    df['temp_change'] = df['temperature'].diff()
    
    # Moving averages
    df['temp_ma_6h'] = df['temperature'].rolling(6).mean()
    
    # Interaction terms
    df['temp_humidity'] = df['temperature'] * df['humidity']
    
    return df
```

#### Time Features
```python
def create_time_features(df):
    """Create advanced time features"""
    # Cyclical time
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    
    # Special periods
    df['is_peak'] = df.index.hour.isin([9, 10, 11, 17, 18, 19])
    
    return df
```

### 3. Advanced Optimization

#### Custom Constraints
```python
def add_custom_constraints(model, data):
    """Add custom optimization constraints"""
    
    # Ramp rate constraints
    def ramp_rate_rule(model, t):
        if t == 0:
            return Constraint.Skip
        return (
            abs(model.production[t] - model.production[t-1]) 
            <= model.max_ramp_rate
        )
    
    model.RampConstraint = Constraint(
        model.time_steps,
        rule=ramp_rate_rule
    )
```

#### Multi-Objective Optimization
```python
def create_multi_objective(model, data):
    """Create multi-objective optimization"""
    
    # Cost objective
    def cost_objective(model):
        return sum(
            model.production[t] * model.cost[t] 
            for t in model.time_steps
        )
    
    # Emissions objective
    def emission_objective(model):
        return sum(
            model.production[t] * model.emission_factor[t] 
            for t in model.time_steps
        )
    
    model.Cost = Objective(rule=cost_objective)
    model.Emissions = Objective(rule=emission_objective)
```

### 4. Performance Optimization

#### Parallel Processing
```python
from concurrent.futures import ProcessPoolExecutor

def parallel_forecast(cities, dates):
    """Generate forecasts in parallel"""
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(generate_forecast, city, date)
            for city, date in zip(cities, dates)
        ]
        results = [f.result() for f in futures]
    return results
```

#### Batch Processing
```python
def batch_process(data, batch_size=1000):
    """Process data in batches"""
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        process_batch(batch)
```

### 5. Advanced Analytics

#### Uncertainty Quantification
```python
def calculate_uncertainty(predictions, confidence=0.95):
    """Calculate prediction intervals"""
    lower = np.percentile(predictions, (1 - confidence) / 2 * 100)
    upper = np.percentile(predictions, (1 + confidence) / 2 * 100)
    return lower, upper
```

#### Sensitivity Analysis
```python
def sensitivity_analysis(model, data, features):
    """Perform sensitivity analysis"""
    sensitivities = {}
    for feature in features:
        perturbed = data.copy()
        perturbed[feature] *= 1.1  # 10% increase
        
        baseline = model.predict(data)
        perturbed_pred = model.predict(perturbed)
        
        sensitivity = (perturbed_pred - baseline) / baseline
        sensitivities[feature] = sensitivity.mean()
    
    return sensitivities
```

### 6. Custom Metrics

#### Define Metrics
```python
def weighted_mape(y_true, y_pred, weights):
    """Calculate weighted MAPE"""
    return np.average(
        np.abs((y_true - y_pred) / y_true),
        weights=weights
    ) * 100

def peak_accuracy(y_true, y_pred, peak_hours):
    """Calculate accuracy during peak hours"""
    peak_mask = peak_hours
    peak_true = y_true[peak_mask]
    peak_pred = y_pred[peak_mask]
    return mean_absolute_error(peak_true, peak_pred)
```

### 7. Advanced Visualization

#### Custom Plots
```python
def plot_energy_mix(forecast):
    """Create energy mix visualization"""
    fig = go.Figure()
    
    for source in ['solar', 'wind', 'hydro']:
        fig.add_trace(
            go.Area(
                name=source,
                x=forecast.index,
                y=forecast[source],
                stackgroup='one'
            )
        )
    
    fig.update_layout(
        title='Energy Mix Over Time',
        xaxis_title='Time',
        yaxis_title='Energy (MWh)'
    )
    
    return fig
```

### 8. API Extensions

#### Custom Endpoints
```python
@router.post("/forecast/custom")
async def custom_forecast(
    request: CustomForecastRequest,
    background_tasks: BackgroundTasks
):
    """Generate custom forecast"""
    # Process request
    result = await process_custom_forecast(request)
    
    # Schedule background task
    background_tasks.add_task(
        save_forecast_results,
        result
    )
    
    return result
```

### 9. Error Handling

#### Custom Exceptions
```python
class ForecastError(Exception):
    """Base class for forecast errors"""
    pass

class DataQualityError(ForecastError):
    """Raised when data quality is insufficient"""
    pass

class ModelError(ForecastError):
    """Raised when model encounters an error"""
    pass
```

### 10. Monitoring

#### Custom Metrics
```python
from prometheus_client import Gauge, Histogram

MODEL_ACCURACY = Gauge(
    'model_accuracy',
    'Model prediction accuracy',
    ['city', 'model']
)

FORECAST_TIME = Histogram(
    'forecast_generation_seconds',
    'Time spent generating forecast',
    ['city']
)
```

## Best Practices

### 1. Code Organization
- Use clear module structure
- Follow consistent naming
- Document complex logic
- Write unit tests

### 2. Performance
- Profile code regularly
- Use appropriate data structures
- Implement caching
- Optimize database queries

### 3. Maintenance
- Monitor system health
- Log important events
- Update dependencies
- Review security

## Troubleshooting

### Common Issues
1. Memory usage
2. Slow predictions
3. Data quality
4. Optimization failures

### Solutions
1. Implement batching
2. Use caching
3. Add validation
4. Debug optimization

## Support
- GitHub Issues
- Documentation
- Email support
- Community forums
