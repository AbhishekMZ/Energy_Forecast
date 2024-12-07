"""Test configuration and fixtures"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from fastapi.testclient import TestClient

from energy_forecast.api.main import app
from energy_forecast.core.models.energy_forecast_model import EnergyForecastModel

@pytest.fixture
def test_client():
    """Create test client"""
    return TestClient(app)

@pytest.fixture
def mock_api_key():
    """Mock API key for testing"""
    return "test_key_123"

@pytest.fixture
def sample_weather_data():
    """Generate sample weather data"""
    dates = pd.date_range(
        start=datetime.now(),
        periods=24,
        freq='H'
    )
    
    return pd.DataFrame({
        'temperature': np.random.uniform(20, 35, 24),
        'humidity': np.random.uniform(40, 80, 24),
        'cloud_cover': np.random.uniform(0, 100, 24),
        'solar_radiation': np.random.uniform(0, 1000, 24),
        'wind_speed': np.random.uniform(0, 20, 24),
        'precipitation': np.random.uniform(0, 10, 24)
    }, index=dates)

@pytest.fixture
def sample_forecast_request():
    """Generate sample forecast request"""
    return {
        "city": "Mumbai",
        "start_date": datetime.now().isoformat(),
        "end_date": (datetime.now() + timedelta(days=1)).isoformat(),
        "weather_forecast": [{
            "temperature": 30.0,
            "humidity": 60.0,
            "cloud_cover": 50.0,
            "solar_radiation": 800.0,
            "wind_speed": 10.0,
            "precipitation": 0.0
        }]
    }

@pytest.fixture
def mock_forecast_model():
    """Create mock forecast model"""
    model = EnergyForecastModel("Mumbai")
    return model
