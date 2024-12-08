# Contribution Guide

## Overview

Thank you for considering contributing to the Energy Forecast Platform! This guide will help you understand our development process and standards.

## Code of Conduct

### Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

Examples of behavior that contributes to creating a positive environment include:
- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

## Getting Started

### 1. Development Environment

```bash
# Clone the repository
git clone https://github.com/company/energy-forecast.git
cd energy-forecast

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements-dev.txt
```

### 2. Code Style

We follow PEP 8 with these additional rules:

```python
# Good example
def calculate_energy_consumption(
    temperature: float,
    humidity: float,
    time_of_day: datetime
) -> float:
    """
    Calculate energy consumption based on environmental factors.
    
    Args:
        temperature: Current temperature in Celsius
        humidity: Current humidity percentage
        time_of_day: Timestamp of prediction
    
    Returns:
        Predicted energy consumption in kWh
    """
    # Implementation
```

## Development Process

### 1. Branching Strategy

```bash
# Feature branch
git checkout -b feature/your-feature-name

# Bug fix branch
git checkout -b fix/bug-description

# Documentation branch
git checkout -b docs/documentation-description
```

### 2. Commit Messages

Follow the Conventional Commits specification:

```
feat: add temperature normalization
fix: correct database connection timeout
docs: update API documentation
test: add unit tests for prediction endpoint
refactor: optimize data preprocessing
```

## Testing Requirements

### 1. Unit Tests

```python
# tests/unit/test_model.py
def test_model_prediction():
    """Test model prediction accuracy."""
    model = EnergyModel()
    
    # Test data
    input_data = {
        'temperature': 25.0,
        'humidity': 60.0,
        'time': datetime.now()
    }
    
    prediction = model.predict(input_data)
    
    assert isinstance(prediction, float)
    assert 0 <= prediction <= 1000  # Valid range
```

### 2. Integration Tests

```python
# tests/integration/test_api.py
def test_prediction_endpoint():
    """Test prediction API endpoint."""
    response = client.post(
        '/api/v1/predict',
        json={
            'city_id': 1,
            'timestamp': '2024-01-01T00:00:00Z'
        }
    )
    
    assert response.status_code == 200
    assert 'prediction' in response.json()
```

## Documentation Requirements

### 1. Code Documentation

```python
class EnergyModel:
    """
    Energy consumption prediction model.
    
    This class implements the ensemble model combining LSTM,
    XGBoost, and Transformer models for energy prediction.
    
    Attributes:
        models (Dict[str, BaseModel]): Dictionary of underlying models
        scaler (StandardScaler): Data standardization
        config (Dict[str, Any]): Model configuration
    """
    
    def predict(
        self,
        features: Dict[str, Any]
    ) -> float:
        """
        Make energy consumption prediction.
        
        Args:
            features: Dictionary of input features
        
        Returns:
            Predicted energy consumption in kWh
        
        Raises:
            ValueError: If features are invalid
        """
```

### 2. API Documentation

```python
@app.post("/api/v1/predict")
async def predict_consumption(
    request: PredictionRequest
) -> PredictionResponse:
    """
    Predict energy consumption.
    
    Args:
        request: Prediction request containing city_id and timestamp
    
    Returns:
        Prediction response with energy consumption forecast
    
    Raises:
        HTTPException: If request is invalid or prediction fails
    """
```

## Pull Request Process

### 1. PR Template

```markdown
## Description
Brief description of the changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] All tests passing

## Documentation
- [ ] Code comments updated
- [ ] API documentation updated
- [ ] README updated if needed
```

### 2. Review Process

1. Code Review Requirements:
   - At least one approval from core team
   - All tests passing
   - Code style compliance
   - Documentation updated

2. Review Response Time:
   - First review within 2 business days
   - Follow-up reviews within 1 business day

## Release Process

### 1. Version Bumping

```bash
# Update version in setup.py
sed -i 's/version=.*/version="1.2.3"/' setup.py

# Update changelog
echo "## [1.2.3] - $(date +%Y-%m-%d)" >> CHANGELOG.md
```

### 2. Release Checklist

```markdown
## Pre-release
- [ ] Version bumped
- [ ] Changelog updated
- [ ] Tests passing
- [ ] Documentation updated
- [ ] Performance verified
- [ ] Security scan completed

## Release
- [ ] Tag created
- [ ] Release notes published
- [ ] Artifacts uploaded
- [ ] Deployment verified

## Post-release
- [ ] Monitor metrics
- [ ] Update documentation
- [ ] Notify stakeholders
```

## Best Practices

### 1. Code Quality

- Write self-documenting code
- Follow SOLID principles
- Keep functions small and focused
- Use meaningful variable names
- Add appropriate error handling

### 2. Performance

- Profile code before optimization
- Use appropriate data structures
- Implement caching where beneficial
- Optimize database queries
- Consider resource usage

### 3. Security

- Validate all inputs
- Use parameterized queries
- Follow least privilege principle
- Keep dependencies updated
- Implement proper error handling

## Additional Resources

- [Development Setup](./development_setup.md)
- [API Reference](./api_reference.md)
- [Testing Guide](./testing_guide.md)
- [Style Guide](./style_guide.md)
- [Security Guidelines](./security_guide.md)
