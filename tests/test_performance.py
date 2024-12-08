"""Performance tests for the Energy Forecast Platform."""

import pytest
import asyncio
import aiohttp
import time
import statistics
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import numpy as np
from energy_forecast.core.utils.performance_monitoring import PerformanceMonitor
from energy_forecast.models.training_pipeline import ModelTrainingPipeline
from energy_forecast.database.connection import DatabaseConnection

class TestPerformance:
    """Performance test suite."""

    @pytest.fixture(scope="class")
    def performance_monitor(self):
        return PerformanceMonitor()

    @pytest.fixture(scope="class")
    def model_pipeline(self):
        return ModelTrainingPipeline()

    @pytest.fixture(scope="class")
    def db_connection(self):
        return DatabaseConnection()

    async def test_api_latency(self, performance_monitor):
        """Test API endpoint latency under various conditions."""
        async with aiohttp.ClientSession() as session:
            latencies = []
            
            for _ in range(100):
                start_time = time.time()
                async with session.get(
                    "http://localhost:8000/api/v1/forecast",
                    params={
                        "city": "Mumbai",
                        "date": datetime.now().strftime("%Y-%m-%d")
                    }
                ) as response:
                    assert response.status == 200
                    latency = time.time() - start_time
                    latencies.append(latency)

            # Analyze latencies
            avg_latency = statistics.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)

            assert avg_latency < 0.1  # 100ms average
            assert p95_latency < 0.2  # 200ms 95th percentile
            assert p99_latency < 0.5  # 500ms 99th percentile

    async def test_concurrent_performance(self):
        """Test system performance under concurrent load."""
        async with aiohttp.ClientSession() as session:
            # Generate concurrent requests
            concurrent_requests = 50
            tasks = []
            
            for i in range(concurrent_requests):
                task = session.get(
                    "http://localhost:8000/api/v1/forecast",
                    params={
                        "city": "Mumbai",
                        "date": datetime.now().strftime("%Y-%m-%d")
                    }
                )
                tasks.append(task)

            # Execute concurrent requests
            start_time = time.time()
            responses = await asyncio.gather(*tasks)
            total_time = time.time() - start_time

            # Verify responses
            assert all(r.status == 200 for r in responses)
            assert total_time < 5  # All requests complete within 5 seconds

    def test_database_performance(self, db_connection, performance_monitor):
        """Test database performance under load."""
        with performance_monitor as monitor:
            with db_connection.get_session() as session:
                # Perform multiple database operations
                for _ in range(1000):
                    session.execute(
                        "SELECT * FROM predictions ORDER BY id DESC LIMIT 1"
                    )

        metrics = monitor.get_metrics()
        assert metrics["avg_query_time"] < 0.01  # 10ms per query average

    def test_model_inference_performance(self, model_pipeline):
        """Test model inference performance."""
        test_data = self._generate_test_data(1000)
        
        with PerformanceMonitor() as monitor:
            predictions = model_pipeline.batch_predict(test_data)

        metrics = monitor.get_metrics()
        assert metrics["inference_time"] / len(test_data) < 0.05  # 50ms per prediction

    def test_cache_performance(self, performance_monitor):
        """Test cache hit rates and latency."""
        with performance_monitor as monitor:
            for _ in range(100):
                # Make same request to test cache
                self._make_cached_request()

        metrics = monitor.get_metrics()
        assert metrics["cache_hit_rate"] > 0.95  # 95% cache hit rate
        assert metrics["cache_latency"] < 0.005  # 5ms cache latency

    def test_memory_usage(self, model_pipeline):
        """Test memory usage under load."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Perform memory-intensive operations
        large_batch = self._generate_test_data(10000)
        predictions = model_pipeline.batch_predict(large_batch)

        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB

        assert memory_increase < 500  # Less than 500MB increase

    def test_model_training_performance(self, model_pipeline):
        """Test model training performance metrics."""
        training_data = self._generate_training_data()
        
        with PerformanceMonitor() as monitor:
            model_pipeline.train_model(
                training_data,
                epochs=10,
                batch_size=32
            )

        metrics = monitor.get_metrics()
        assert metrics["training_time"] < 3600  # Training completes within 1 hour
        assert metrics["model_accuracy"] > 0.8  # Model achieves 80% accuracy

    def test_data_processing_pipeline(self, performance_monitor):
        """Test data processing pipeline performance."""
        raw_data = self._generate_raw_data(10000)
        
        with performance_monitor as monitor:
            processed_data = self._process_data_pipeline(raw_data)

        metrics = monitor.get_metrics()
        assert metrics["processing_time"] / len(raw_data) < 0.01  # 10ms per record

    def test_api_throughput(self):
        """Test API throughput capacity."""
        async def make_requests(num_requests):
            async with aiohttp.ClientSession() as session:
                tasks = [
                    session.get(
                        "http://localhost:8000/api/v1/forecast",
                        params={
                            "city": "Mumbai",
                            "date": datetime.now().strftime("%Y-%m-%d")
                        }
                    )
                    for _ in range(num_requests)
                ]
                return await asyncio.gather(*tasks)

        # Test different throughput levels
        throughputs = [10, 50, 100, 200]
        for throughput in throughputs:
            start_time = time.time()
            responses = asyncio.run(make_requests(throughput))
            total_time = time.time() - start_time
            
            requests_per_second = throughput / total_time
            assert requests_per_second >= throughput * 0.8  # Maintain 80% of target throughput

    # Helper methods
    def _generate_test_data(self, size):
        """Generate test data for performance testing."""
        return [
            {
                "city": "Mumbai",
                "date": (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d"),
                "features": np.random.rand(10)
            }
            for i in range(size)
        ]

    def _generate_training_data(self):
        """Generate training data for model performance testing."""
        X = np.random.rand(10000, 10)
        y = np.random.rand(10000)
        return X, y

    def _generate_raw_data(self, size):
        """Generate raw data for pipeline testing."""
        return [
            {
                "timestamp": datetime.now().isoformat(),
                "measurements": list(np.random.rand(20)),
                "metadata": {
                    "sensor_id": f"sensor_{i}",
                    "location": "Mumbai"
                }
            }
            for i in range(size)
        ]

    def _process_data_pipeline(self, raw_data):
        """Simulate data processing pipeline."""
        processed_data = []
        for record in raw_data:
            # Simulate processing steps
            processed_record = {
                "timestamp": datetime.fromisoformat(record["timestamp"]),
                "features": np.array(record["measurements"]),
                "metadata": record["metadata"]
            }
            processed_data.append(processed_record)
        return processed_data

    def _make_cached_request(self):
        """Make a request that should be cached."""
        import requests
        return requests.get(
            "http://localhost:8000/api/v1/forecast",
            params={
                "city": "Mumbai",
                "date": datetime.now().strftime("%Y-%m-%d")
            }
        )
