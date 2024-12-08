"""Performance monitoring utilities for API and model performance."""
import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional
from prometheus_client import Counter, Histogram, Gauge, Summary
from .logger import StructuredLogger

logger = StructuredLogger(__name__)

# Request metrics
REQUEST_LATENCY = Histogram(
    'request_latency_seconds',
    'Request latency in seconds',
    ['endpoint', 'method', 'status']
)

REQUEST_COUNT = Counter(
    'request_total',
    'Total number of requests',
    ['endpoint', 'method', 'status']
)

# Model metrics
MODEL_INFERENCE_TIME = Histogram(
    'model_inference_seconds',
    'Model inference time in seconds',
    ['model_name', 'batch_size']
)

MODEL_INFERENCE_COUNT = Counter(
    'model_inference_total',
    'Total number of model inferences',
    ['model_name', 'batch_size']
)

# Cache metrics
CACHE_HIT_RATIO = Gauge(
    'cache_hit_ratio',
    'Cache hit ratio'
)

CACHE_OPERATION_LATENCY = Histogram(
    'cache_operation_seconds',
    'Cache operation latency in seconds',
    ['operation']
)

# Database metrics
DB_QUERY_LATENCY = Histogram(
    'db_query_seconds',
    'Database query latency in seconds',
    ['query_type']
)

DB_CONNECTION_POOL = Gauge(
    'db_connection_pool_size',
    'Database connection pool size'
)

# Batch processing metrics
BATCH_SIZE = Histogram(
    'batch_size',
    'Request batch sizes'
)

BATCH_PROCESSING_TIME = Histogram(
    'batch_processing_seconds',
    'Batch processing time in seconds'
)

class PerformanceProfiler:
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.logger = StructuredLogger(service_name)
        
    def profile_endpoint(self, endpoint_name: str):
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs) -> Any:
                start_time = time.time()
                status = "success"
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    status = "error"
                    raise e
                finally:
                    duration = time.time() - start_time
                    REQUEST_LATENCY.labels(
                        endpoint=endpoint_name,
                        method=func.__name__,
                        status=status
                    ).observe(duration)
                    REQUEST_COUNT.labels(
                        endpoint=endpoint_name,
                        method=func.__name__,
                        status=status
                    ).inc()
                    
                    self.logger.info(
                        f"API request processed",
                        endpoint=endpoint_name,
                        duration=duration,
                        status=status
                    )
            return wrapper
        return decorator
    
    def profile_model(self, model_name: str):
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs) -> Any:
                start_time = time.time()
                batch_size = len(args[1]) if len(args) > 1 else 1
                try:
                    result = await func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    MODEL_INFERENCE_TIME.labels(
                        model_name=model_name,
                        batch_size=batch_size
                    ).observe(duration)
                    MODEL_INFERENCE_COUNT.labels(
                        model_name=model_name,
                        batch_size=batch_size
                    ).inc()
                    
                    self.logger.info(
                        f"Model inference completed",
                        model=model_name,
                        batch_size=batch_size,
                        duration=duration
                    )
                    return result
                except Exception as e:
                    self.logger.error(
                        f"Model inference failed",
                        error=e,
                        model=model_name,
                        batch_size=batch_size
                    )
                    raise
            return wrapper
        return decorator
    
    def profile_cache(self, operation: str):
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs) -> Any:
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    CACHE_OPERATION_LATENCY.labels(
                        operation=operation
                    ).observe(duration)
                    
                    if operation == "get":
                        # Update cache hit ratio
                        hit = result is not None
                        current_ratio = CACHE_HIT_RATIO._value.get()
                        new_ratio = (current_ratio + (1 if hit else 0)) / 2
                        CACHE_HIT_RATIO.set(new_ratio)
                        
                        self.logger.info(
                            f"Cache {operation} operation",
                            hit=hit,
                            duration=duration
                        )
                    
                    return result
                except Exception as e:
                    self.logger.error(
                        f"Cache operation failed",
                        error=e,
                        operation=operation
                    )
                    raise
            return wrapper
        return decorator
    
    def profile_db(self, query_type: str):
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs) -> Any:
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    DB_QUERY_LATENCY.labels(
                        query_type=query_type
                    ).observe(duration)
                    
                    self.logger.info(
                        f"Database query executed",
                        query_type=query_type,
                        duration=duration
                    )
                    return result
                except Exception as e:
                    self.logger.error(
                        f"Database query failed",
                        error=e,
                        query_type=query_type
                    )
                    raise
            return wrapper
        return decorator
    
    def profile_batch(self):
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs) -> Any:
                start_time = time.time()
                batch_size = len(args[1]) if len(args) > 1 else 1
                
                try:
                    result = await func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    BATCH_SIZE.observe(batch_size)
                    BATCH_PROCESSING_TIME.observe(duration)
                    
                    self.logger.info(
                        f"Batch processing completed",
                        batch_size=batch_size,
                        duration=duration
                    )
                    return result
                except Exception as e:
                    self.logger.error(
                        f"Batch processing failed",
                        error=e,
                        batch_size=batch_size
                    )
                    raise
            return wrapper
        return decorator

# Initialize global profiler
profiler = PerformanceProfiler("energy_forecast")

class BatchProcessor:
    """Process requests in batches for improved performance."""
    
    def __init__(
        self,
        max_batch_size: int = 32,
        max_wait_time: float = 0.1
    ):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.current_batch: List = []
        self.last_process_time = time.time()

    async def add_to_batch(self, item) -> bool:
        """Add item to current batch and return True if batch should be processed."""
        self.current_batch.append(item)
        
        current_time = time.time()
        should_process = (
            len(self.current_batch) >= self.max_batch_size or
            current_time - self.last_process_time >= self.max_wait_time
        )
        
        if should_process:
            self.last_process_time = current_time
            
        return should_process

    @profiler.profile_batch()
    async def process_batch(self, processor_func):
        """Process the current batch using the provided function."""
        if not self.current_batch:
            return []
            
        try:
            results = await processor_func(self.current_batch)
            self.current_batch = []
            return results
        except Exception as e:
            logger.error(f"Batch processing error: {str(e)}")
            self.current_batch = []
            raise

class ModelProfiler:
    """Profile model inference performance."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.inference_times: List[float] = []
        self.batch_sizes: List[int] = []

    @profiler.profile_model(model_name)
    async def profile_inference(self, batch_size: int = 1):
        """Decorator to profile model inference."""
        pass

    def get_performance_stats(self) -> Dict:
        """Get performance statistics for the model."""
        if not self.inference_times:
            return {}
            
        return {
            "model_name": self.model_name,
            "avg_inference_time": np.mean(self.inference_times),
            "p95_inference_time": np.percentile(self.inference_times, 95),
            "min_inference_time": np.min(self.inference_times),
            "max_inference_time": np.max(self.inference_times),
            "avg_batch_size": np.mean(self.batch_sizes),
            "total_inferences": len(self.inference_times)
        }

class CacheProfiler:
    """Profile cache performance."""
    
    def __init__(self, cache_name: str):
        self.cache_name = cache_name
        self.hits = 0
        self.misses = 0

    @profiler.profile_cache("get")
    async def get(self, key):
        """Get item from cache."""
        pass

    @profiler.profile_cache("set")
    async def set(self, key, value):
        """Set item in cache."""
        pass

    def record_cache_access(self, hit: bool):
        """Record a cache hit or miss."""
        if hit:
            self.hits += 1
            CACHE_HITS.labels(cache_type=self.cache_name).inc()
        else:
            self.misses += 1
            CACHE_MISSES.labels(cache_type=self.cache_name).inc()

    def get_stats(self) -> Dict:
        """Get cache performance statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        return {
            "cache_name": self.cache_name,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "total_accesses": total
        }
