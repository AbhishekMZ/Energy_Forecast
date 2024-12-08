"""API endpoint tests."""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
import json
from energy_forecast.main import app
from energy_forecast.core.utils.performance_monitoring import PerformanceMonitor

client = TestClient(app)

@pytest.fixture
def test_data():
    return {
        "city": "Mumbai",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "include_weather": True
    }

@pytest.fixture
def auth_headers():
    return {"Authorization": f"Bearer {pytest.API_KEY}"}

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_forecast_endpoint(test_data, auth_headers):
    response = client.get(
        "/api/v1/forecast",
        params=test_data,
        headers=auth_headers
    )
    assert response.status_code == 200
    data = response.json()
    assert "forecast" in data
    assert "consumption_kwh" in data["forecast"]
    assert "confidence" in data["forecast"]

def test_batch_forecast(auth_headers):
    payload = {
        "locations": [
            {"city": "Mumbai", "date": datetime.now().strftime("%Y-%m-%d")},
            {"city": "Delhi", "date": datetime.now().strftime("%Y-%m-%d")}
        ]
    }
    response = client.post(
        "/api/v1/batch_forecast",
        json=payload,
        headers=auth_headers
    )
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 2

def test_invalid_auth():
    response = client.get("/api/v1/forecast")
    assert response.status_code == 401

def test_invalid_city(auth_headers):
    params = {
        "city": "InvalidCity",
        "date": datetime.now().strftime("%Y-%m-%d")
    }
    response = client.get(
        "/api/v1/forecast",
        params=params,
        headers=auth_headers
    )
    assert response.status_code == 400

def test_rate_limiting(auth_headers):
    for _ in range(101):  # Exceed rate limit
        client.get(
            "/api/v1/forecast",
            params={"city": "Mumbai", "date": datetime.now().strftime("%Y-%m-%d")},
            headers=auth_headers
        )
    response = client.get(
        "/api/v1/forecast",
        params={"city": "Mumbai", "date": datetime.now().strftime("%Y-%m-%d")},
        headers=auth_headers
    )
    assert response.status_code == 429

def test_cache_behavior(test_data, auth_headers):
    # First request
    response1 = client.get(
        "/api/v1/forecast",
        params=test_data,
        headers=auth_headers
    )
    
    # Second request (should be cached)
    response2 = client.get(
        "/api/v1/forecast",
        params=test_data,
        headers=auth_headers
    )
    
    assert response1.json() == response2.json()
    assert "X-Cache-Hit" in response2.headers

def test_error_handling(auth_headers):
    response = client.get(
        "/api/v1/forecast",
        params={"city": "Mumbai"},  # Missing date parameter
        headers=auth_headers
    )
    assert response.status_code == 400
    assert "error" in response.json()

def test_performance_monitoring():
    with PerformanceMonitor() as monitor:
        client.get("/health")
    
    metrics = monitor.get_metrics()
    assert "request_duration" in metrics
    assert metrics["request_duration"] > 0

def test_model_versioning(auth_headers):
    # Get predictions from two different model versions
    headers = {**auth_headers, "X-Model-Version": "v1"}
    response1 = client.get(
        "/api/v1/forecast",
        params={"city": "Mumbai", "date": datetime.now().strftime("%Y-%m-%d")},
        headers=headers
    )
    
    headers = {**auth_headers, "X-Model-Version": "v2"}
    response2 = client.get(
        "/api/v1/forecast",
        params={"city": "Mumbai", "date": datetime.now().strftime("%Y-%m-%d")},
        headers=headers
    )
    
    # Ensure different versions return different results
    assert response1.json()["model_version"] == "v1"
    assert response2.json()["model_version"] == "v2"

def test_batch_size_limits(auth_headers):
    # Test with too many locations
    payload = {
        "locations": [
            {"city": "Mumbai", "date": datetime.now().strftime("%Y-%m-%d")}
            for _ in range(101)  # Exceed batch size limit
        ]
    }
    response = client.post(
        "/api/v1/batch_forecast",
        json=payload,
        headers=auth_headers
    )
    assert response.status_code == 400
    assert "batch size" in response.json()["error"].lower()

def test_monitoring_endpoints(auth_headers):
    response = client.get(
        "/metrics",
        headers=auth_headers
    )
    assert response.status_code == 200
    assert "request_latency" in response.text
    assert "model_inference_time" in response.text

def test_security_headers():
    response = client.get("/health")
    assert "X-Content-Type-Options" in response.headers
    assert "X-Frame-Options" in response.headers
    assert "X-XSS-Protection" in response.headers
