"""Integration tests for the Energy Forecast Platform."""

import pytest
from fastapi.testclient import TestClient
import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
from energy_forecast.main import app
from energy_forecast.core.utils.performance_monitoring import PerformanceMonitor
from energy_forecast.models.training_pipeline import ModelTrainingPipeline
from energy_forecast.database.connection import DatabaseConnection

@pytest.fixture(scope="module")
def test_app():
    client = TestClient(app)
    return client

@pytest.fixture(scope="module")
def db():
    return DatabaseConnection()

@pytest.fixture(scope="module")
def model_pipeline():
    return ModelTrainingPipeline()

class TestSystemIntegration:
    """System-wide integration tests."""

    async def test_concurrent_requests(self, test_app):
        """Test system behavior under concurrent load."""
        async with aiohttp.ClientSession() as session:
            tasks = []
            for _ in range(50):
                tasks.append(
                    session.get(
                        "http://localhost:8000/api/v1/forecast",
                        params={
                            "city": "Mumbai",
                            "date": datetime.now().strftime("%Y-%m-%d")
                        }
                    )
                )
            responses = await asyncio.gather(*tasks)
            assert all(r.status == 200 for r in responses)

    def test_data_consistency(self, test_app, db):
        """Test data consistency across system components."""
        # Make prediction
        response = test_app.post(
            "/api/v1/forecast",
            json={
                "city": "Mumbai",
                "date": datetime.now().strftime("%Y-%m-%d")
            }
        )
        prediction_id = response.json()["prediction_id"]

        # Verify database record
        with db.get_session() as session:
            record = session.execute(
                "SELECT * FROM predictions WHERE id = :id",
                {"id": prediction_id}
            ).fetchone()
            assert record is not None
            assert record.status == "completed"

    def test_error_propagation(self, test_app):
        """Test error handling across components."""
        # Test database connection error
        with pytest.raises(Exception):
            test_app.get(
                "/api/v1/forecast",
                params={"city": "Mumbai", "date": "invalid_date"}
            )

        # Test model error
        response = test_app.post(
            "/api/v1/forecast",
            json={"city": "InvalidCity"}
        )
        assert response.status_code == 400
        assert "error" in response.json()

    def test_cache_consistency(self, test_app):
        """Test cache consistency with database."""
        params = {
            "city": "Mumbai",
            "date": datetime.now().strftime("%Y-%m-%d")
        }

        # First request
        response1 = test_app.get("/api/v1/forecast", params=params)
        
        # Second request (should be cached)
        response2 = test_app.get("/api/v1/forecast", params=params)
        
        # Verify cache hit
        assert response1.json() == response2.json()
        assert "X-Cache-Hit" in response2.headers

        # Verify database consistency
        with db.get_session() as session:
            cache_record = session.execute(
                "SELECT * FROM cache_metrics ORDER BY id DESC LIMIT 1"
            ).fetchone()
            assert cache_record.hit_count > 0

class TestPerformanceIntegration:
    """Performance-focused integration tests."""

    def test_batch_processing_performance(self, test_app):
        """Test batch processing under load."""
        batch_sizes = [10, 50, 100]
        for size in batch_sizes:
            payload = {
                "locations": [
                    {
                        "city": "Mumbai",
                        "date": datetime.now().strftime("%Y-%m-%d")
                    }
                    for _ in range(size)
                ]
            }
            with PerformanceMonitor() as monitor:
                response = test_app.post("/api/v1/batch_forecast", json=payload)
            
            metrics = monitor.get_metrics()
            assert metrics["batch_processing_time"] < size * 0.1  # 100ms per item max
            assert response.status_code == 200

    def test_model_retraining_performance(self, model_pipeline):
        """Test model retraining performance."""
        with PerformanceMonitor() as monitor:
            model_pipeline.retrain_model(
                training_data=get_test_data(),
                model_params={"batch_size": 32}
            )
        
        metrics = monitor.get_metrics()
        assert metrics["training_time"] < 300  # 5 minutes max
        assert metrics["model_accuracy"] > 0.8

    def test_database_performance(self, db):
        """Test database performance under load."""
        with PerformanceMonitor() as monitor:
            with db.get_session() as session:
                # Perform multiple queries
                for _ in range(100):
                    session.execute(
                        "SELECT * FROM predictions ORDER BY id DESC LIMIT 1"
                    )
        
        metrics = monitor.get_metrics()
        assert metrics["avg_query_time"] < 0.1  # 100ms max per query

class TestSecurityIntegration:
    """Security-focused integration tests."""

    def test_authentication_flow(self, test_app):
        """Test complete authentication flow."""
        # Test invalid token
        response = test_app.get(
            "/api/v1/forecast",
            headers={"Authorization": "Bearer invalid_token"}
        )
        assert response.status_code == 401

        # Test expired token
        expired_token = create_expired_token()
        response = test_app.get(
            "/api/v1/forecast",
            headers={"Authorization": f"Bearer {expired_token}"}
        )
        assert response.status_code == 401

        # Test valid token
        valid_token = create_valid_token()
        response = test_app.get(
            "/api/v1/forecast",
            headers={"Authorization": f"Bearer {valid_token}"}
        )
        assert response.status_code == 200

    def test_rate_limiting(self, test_app):
        """Test rate limiting across endpoints."""
        valid_token = create_valid_token()
        headers = {"Authorization": f"Bearer {valid_token}"}

        # Make requests up to limit
        for _ in range(100):
            response = test_app.get(
                "/api/v1/forecast",
                headers=headers
            )
            assert response.status_code == 200

        # Verify rate limit
        response = test_app.get("/api/v1/forecast", headers=headers)
        assert response.status_code == 429

    def test_data_validation(self, test_app):
        """Test input validation and sanitization."""
        valid_token = create_valid_token()
        headers = {"Authorization": f"Bearer {valid_token}"}

        # Test SQL injection attempt
        response = test_app.get(
            "/api/v1/forecast",
            params={"city": "Mumbai'; DROP TABLE predictions; --"},
            headers=headers
        )
        assert response.status_code == 400

        # Test XSS attempt
        response = test_app.get(
            "/api/v1/forecast",
            params={"city": "<script>alert('xss')</script>"},
            headers=headers
        )
        assert response.status_code == 400

class TestFailureRecovery:
    """Failure recovery and resilience tests."""

    def test_database_failover(self, test_app, db):
        """Test system behavior during database failover."""
        # Simulate database connection failure
        db.engine.dispose()
        
        # System should use cache
        response = test_app.get(
            "/api/v1/forecast",
            params={
                "city": "Mumbai",
                "date": datetime.now().strftime("%Y-%m-%d")
            }
        )
        assert response.status_code == 200
        assert "X-Failover-Cache" in response.headers

    def test_cache_failure_recovery(self, test_app):
        """Test system behavior during cache failure."""
        # Simulate cache failure
        app.state.cache_client.flushall()
        
        # System should fall back to database
        response = test_app.get(
            "/api/v1/forecast",
            params={
                "city": "Mumbai",
                "date": datetime.now().strftime("%Y-%m-%d")
            }
        )
        assert response.status_code == 200
        assert "X-Database-Fallback" in response.headers

    def test_model_version_fallback(self, test_app):
        """Test model version fallback mechanism."""
        # Request with non-existent model version
        response = test_app.get(
            "/api/v1/forecast",
            params={
                "city": "Mumbai",
                "date": datetime.now().strftime("%Y-%m-%d")
            },
            headers={"X-Model-Version": "non_existent"}
        )
        assert response.status_code == 200
        assert response.json()["model_version"] == "latest"  # Fallback to latest

# Helper functions
def get_test_data():
    """Get test data for model training."""
    # Implementation here
    pass

def create_expired_token():
    """Create an expired JWT token."""
    # Implementation here
    pass

def create_valid_token():
    """Create a valid JWT token."""
    # Implementation here
    pass
