"""Test data validation functionality"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from energy_forecast.api.data_validation import DataValidator
from energy_forecast.core.validation.basic_validator import BasicDataValidator
from energy_forecast.core.validation.statistical_validator import StatisticalValidator

@pytest.fixture
def data_validator():
    """Create data validator instance"""
    return DataValidator()

@pytest.fixture
def invalid_data():
    """Generate invalid test data"""
    return pd.DataFrame({
        'temperature': [-100, 200] * 12,  # Invalid temperatures
        'humidity': [150, -50] * 12,      # Invalid humidity
        'wind_speed': [-10, 500] * 12     # Invalid wind speeds
    })

def test_basic_validation(data_validator, sample_weather_data):
    """Test basic data validation"""
    valid, errors = data_validator.validate_forecast_request(
        sample_weather_data.to_dict(),
        "Mumbai"
    )
    
    assert valid
    assert not errors

def test_invalid_data_validation(data_validator, invalid_data):
    """Test validation with invalid data"""
    valid, errors = data_validator.validate_forecast_request(
        invalid_data.to_dict(),
        "Mumbai"
    )
    
    assert not valid
    assert len(errors) > 0
    assert any("temperature" in error.lower() for error in errors)
    assert any("humidity" in error.lower() for error in errors)

def test_statistical_validation(data_validator, sample_weather_data):
    """Test statistical validation"""
    metrics = data_validator.check_data_quality(sample_weather_data)
    
    assert isinstance(metrics, dict)
    assert 'completeness' in metrics
    assert 'validity' in metrics
    assert 0 <= metrics['completeness'] <= 1
    assert 0 <= metrics['validity'] <= 1

def test_missing_data_validation(data_validator):
    """Test validation with missing data"""
    data = pd.DataFrame({
        'temperature': [25.0, np.nan, 27.0, np.nan],
        'humidity': [60.0, 65.0, np.nan, np.nan]
    })
    
    valid, errors = data_validator.validate_forecast_request(
        data.to_dict(),
        "Mumbai"
    )
    
    assert not valid
    assert any("missing" in error.lower() for error in errors)

def test_duplicate_timestamp_validation(data_validator):
    """Test validation with duplicate timestamps"""
    dates = [datetime.now()] * 4
    data = pd.DataFrame({
        'temperature': [25.0, 26.0, 27.0, 28.0],
        'humidity': [60.0, 65.0, 70.0, 75.0]
    }, index=dates)
    
    valid, errors = data_validator.validate_forecast_request(
        data.to_dict(),
        "Mumbai"
    )
    
    assert not valid
    assert any("duplicate" in error.lower() for error in errors)

def test_city_specific_validation(data_validator, sample_weather_data):
    """Test city-specific validation rules"""
    # Test for Mumbai
    valid_mumbai, errors_mumbai = data_validator.validate_forecast_request(
        sample_weather_data.to_dict(),
        "Mumbai"
    )
    
    # Test for Delhi
    valid_delhi, errors_delhi = data_validator.validate_forecast_request(
        sample_weather_data.to_dict(),
        "Delhi"
    )
    
    assert valid_mumbai != valid_delhi or errors_mumbai != errors_delhi

def test_data_quality_metrics(data_validator, sample_weather_data):
    """Test data quality metrics calculation"""
    metrics = data_validator.check_data_quality(sample_weather_data)
    
    required_metrics = [
        'completeness',
        'timestamp_consistency',
        'value_consistency',
        'validity',
        'timeliness'
    ]
    
    for metric in required_metrics:
        assert metric in metrics
        assert isinstance(metrics[metric], float)
        assert 0 <= metrics[metric] <= 1
