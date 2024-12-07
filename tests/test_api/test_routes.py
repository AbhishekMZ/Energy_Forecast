"""Test API routes"""

import pytest
from fastapi import status
from datetime import datetime, timedelta

def test_health_check(test_client):
    """Test health check endpoint"""
    response = test_client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["status"] == "healthy"

def test_forecast_demand_valid(test_client, mock_api_key, sample_forecast_request):
    """Test valid forecast demand request"""
    headers = {"X-API-Key": mock_api_key}
    response = test_client.post(
        "/forecast/demand",
        headers=headers,
        json=sample_forecast_request
    )
    
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "demand_forecast" in data
    assert "energy_mix" in data
    assert "confidence_intervals" in data

def test_forecast_demand_invalid_city(test_client, mock_api_key):
    """Test forecast demand with invalid city"""
    headers = {"X-API-Key": mock_api_key}
    invalid_request = {
        "city": "InvalidCity",
        "start_date": datetime.now().isoformat(),
        "end_date": (datetime.now() + timedelta(days=1)).isoformat(),
        "weather_forecast": []
    }
    
    response = test_client.post(
        "/forecast/demand",
        headers=headers,
        json=invalid_request
    )
    
    assert response.status_code == status.HTTP_400_BAD_REQUEST

def test_forecast_demand_invalid_dates(test_client, mock_api_key):
    """Test forecast demand with invalid dates"""
    headers = {"X-API-Key": mock_api_key}
    invalid_request = {
        "city": "Mumbai",
        "start_date": (datetime.now() + timedelta(days=1)).isoformat(),
        "end_date": datetime.now().isoformat(),  # end before start
        "weather_forecast": []
    }
    
    response = test_client.post(
        "/forecast/demand",
        headers=headers,
        json=invalid_request
    )
    
    assert response.status_code == status.HTTP_400_BAD_REQUEST

def test_list_cities(test_client, mock_api_key):
    """Test list cities endpoint"""
    headers = {"X-API-Key": mock_api_key}
    response = test_client.get("/forecast/cities", headers=headers)
    
    assert response.status_code == status.HTTP_200_OK
    cities = response.json()
    assert isinstance(cities, list)
    assert "Mumbai" in cities
    assert "Delhi" in cities

def test_validate_data_valid(test_client, mock_api_key, sample_weather_data):
    """Test data validation with valid data"""
    headers = {"X-API-Key": mock_api_key}
    response = test_client.post(
        "/data/validate",
        headers=headers,
        params={"city": "Mumbai"},
        json=sample_weather_data.to_dict()
    )
    
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "valid"
    assert "quality_metrics" in data

def test_validate_data_invalid(test_client, mock_api_key):
    """Test data validation with invalid data"""
    headers = {"X-API-Key": mock_api_key}
    invalid_data = {
        "temperature": [-100] * 24,  # Invalid temperature
        "humidity": [150] * 24       # Invalid humidity
    }
    
    response = test_client.post(
        "/data/validate",
        headers=headers,
        params={"city": "Mumbai"},
        json=invalid_data
    )
    
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "invalid"
    assert len(data["validation_errors"]) > 0
