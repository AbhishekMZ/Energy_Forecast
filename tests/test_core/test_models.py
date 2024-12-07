"""Test core model functionality"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from energy_forecast.core.models.energy_forecast_model import EnergyForecastModel
from energy_forecast.core.models.baseline_models import BaselineModel
from energy_forecast.core.models.advanced_models import AdvancedModel

def test_model_initialization(mock_forecast_model):
    """Test model initialization"""
    assert mock_forecast_model.city == "Mumbai"
    assert hasattr(mock_forecast_model, 'baseline_model')
    assert hasattr(mock_forecast_model, 'advanced_model')

def test_forecast_demand(mock_forecast_model, sample_weather_data):
    """Test demand forecasting"""
    forecast = mock_forecast_model.forecast_demand(
        current_data=sample_weather_data,
        forecast_horizon=24
    )
    
    assert isinstance(forecast, dict)
    assert 'demand_mean' in forecast
    assert 'demand_lower' in forecast
    assert 'demand_upper' in forecast
    assert len(forecast['demand_mean']) == 24

def test_optimize_energy_mix(mock_forecast_model, sample_weather_data):
    """Test energy mix optimization"""
    forecast = mock_forecast_model.forecast_demand(
        current_data=sample_weather_data,
        forecast_horizon=24
    )
    
    energy_mix = mock_forecast_model.optimize_energy_mix(
        forecast,
        sample_weather_data
    )
    
    assert isinstance(energy_mix, dict)
    assert 'energy_mix' in energy_mix
    assert 'schedule' in energy_mix
    assert 'total_cost' in energy_mix

def test_baseline_model_training():
    """Test baseline model training"""
    model = BaselineModel()
    X = pd.DataFrame({
        'temperature': np.random.uniform(20, 35, 100),
        'humidity': np.random.uniform(40, 80, 100)
    })
    y = np.random.uniform(1000, 2000, 100)
    
    model.train(X, y)
    predictions = model.predict(X)
    
    assert len(predictions) == len(y)
    assert isinstance(predictions, np.ndarray)

def test_advanced_model_training():
    """Test advanced model training"""
    model = AdvancedModel()
    X = pd.DataFrame({
        'temperature': np.random.uniform(20, 35, 100),
        'humidity': np.random.uniform(40, 80, 100),
        'cloud_cover': np.random.uniform(0, 100, 100),
        'wind_speed': np.random.uniform(0, 20, 100)
    })
    y = np.random.uniform(1000, 2000, 100)
    
    model.train(X, y)
    predictions = model.predict(X)
    
    assert len(predictions) == len(y)
    assert isinstance(predictions, np.ndarray)

def test_model_cross_validation(mock_forecast_model, sample_weather_data):
    """Test model cross validation"""
    cv_scores = mock_forecast_model.cross_validate(
        X=sample_weather_data,
        y=np.random.uniform(1000, 2000, len(sample_weather_data))
    )
    
    assert isinstance(cv_scores, dict)
    assert 'mae' in cv_scores
    assert 'rmse' in cv_scores
    assert 'r2' in cv_scores

def test_feature_importance(mock_forecast_model, sample_weather_data):
    """Test feature importance calculation"""
    importances = mock_forecast_model.get_feature_importance(
        X=sample_weather_data,
        y=np.random.uniform(1000, 2000, len(sample_weather_data))
    )
    
    assert isinstance(importances, dict)
    assert len(importances) == len(sample_weather_data.columns)
    assert sum(importances.values()) == pytest.approx(1.0)
