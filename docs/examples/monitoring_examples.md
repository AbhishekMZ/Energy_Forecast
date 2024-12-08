# Monitoring and Performance Examples

## Real-World Use Cases

### 1. High API Latency Investigation

#### Scenario
API endpoints are experiencing increased latency during peak hours.

#### Monitoring Approach
```python
# 1. Check API latency metrics
rate(request_latency_seconds_sum[5m]) / rate(request_latency_seconds_count[5m])

# 2. Analyze batch sizes
histogram_quantile(0.95, sum(rate(batch_size_bucket[5m])) by (le))

# 3. Monitor cache performance
cache_hit_ratio
```

#### Example Dashboard Query
```json
{
  "panels": [
    {
      "title": "API Latency Breakdown",
      "targets": [
        {
          "expr": "sum(rate(request_latency_seconds_sum[5m])) by (endpoint) / sum(rate(request_latency_seconds_count[5m])) by (endpoint)",
          "legendFormat": "{{endpoint}}"
        }
      ]
    }
  ]
}
```

#### Resolution Steps
```python
from energy_forecast.core.utils.performance_monitoring import profiler

# 1. Add detailed endpoint profiling
@profiler.profile_endpoint("/api/v1/forecast")
async def get_forecast(city: str, date: str):
    start_time = time.time()
    
    # Cache check with timing
    cached_result = await cache.get(f"forecast:{city}:{date}")
    cache_time = time.time() - start_time
    
    if cached_result:
        logger.info(
            "Cache hit for forecast",
            city=city,
            date=date,
            cache_time=cache_time
        )
        return cached_result
    
    # Model inference with timing
    model_start = time.time()
    result = await model.predict(city, date)
    model_time = time.time() - model_start
    
    logger.info(
        "Model inference completed",
        city=city,
        date=date,
        model_time=model_time,
        total_time=time.time() - start_time
    )
    
    return result
```

### 2. Cache Optimization

#### Scenario
Low cache hit ratio affecting system performance.

#### Monitoring Metrics
```python
# 1. Cache hit ratio trend
rate(cache_hits_total[1h]) / (rate(cache_hits_total[1h]) + rate(cache_misses_total[1h]))

# 2. Cache operation latency
rate(cache_operation_seconds_sum[5m]) / rate(cache_operation_seconds_count[5m])

# 3. Memory usage
redis_memory_used_bytes
```

#### Implementation Example
```python
from energy_forecast.core.utils.caching import CacheManager
from energy_forecast.core.utils.logger import StructuredLogger

logger = StructuredLogger(__name__)

class ForecastCache:
    def __init__(self):
        self.cache = CacheManager()
        self.ttl = 3600  # 1 hour default TTL
        
    @profiler.profile_cache("get")
    async def get_forecast(self, city: str, date: str) -> Optional[dict]:
        """Get forecast from cache with monitoring."""
        cache_key = f"forecast:{city}:{date}"
        
        try:
            result = await self.cache.get(cache_key)
            if result:
                logger.info(
                    "Cache hit for forecast",
                    city=city,
                    date=date,
                    key=cache_key
                )
            else:
                logger.info(
                    "Cache miss for forecast",
                    city=city,
                    date=date,
                    key=cache_key
                )
            return result
        except Exception as e:
            logger.error(
                "Cache error",
                error=str(e),
                city=city,
                date=date
            )
            return None
    
    @profiler.profile_cache("set")
    async def cache_forecast(self, city: str, date: str, data: dict):
        """Cache forecast with TTL."""
        cache_key = f"forecast:{city}:{date}"
        
        try:
            await self.cache.set(
                cache_key,
                data,
                expire=self.ttl
            )
            logger.info(
                "Cached forecast",
                city=city,
                date=date,
                ttl=self.ttl
            )
        except Exception as e:
            logger.error(
                "Cache set error",
                error=str(e),
                city=city,
                date=date
            )
```

### 3. Batch Processing Optimization

#### Scenario
Suboptimal batch sizes causing processing delays.

#### Monitoring Setup
```python
# 1. Batch size distribution
histogram_quantile(0.95, sum(rate(batch_size_bucket[5m])) by (le))

# 2. Processing time per batch
rate(batch_processing_seconds_sum[5m]) / rate(batch_processing_seconds_count[5m])

# 3. Error rate by batch size
sum(rate(batch_error_total[5m])) by (batch_size)
```

#### Implementation Example
```python
from energy_forecast.core.utils.batch_processing import BatchProcessor
from energy_forecast.core.utils.performance_monitoring import profiler

class AdaptiveBatchProcessor:
    def __init__(self):
        self.min_batch_size = 10
        self.max_batch_size = 100
        self.current_batch_size = 50
        self.performance_window = []
        
    @profiler.profile_batch()
    async def process_batch(self, items: List[dict]) -> List[dict]:
        """Process batch with adaptive sizing."""
        start_time = time.time()
        
        try:
            results = await self._process_items(items)
            
            # Record performance
            processing_time = time.time() - start_time
            self.performance_window.append({
                'batch_size': len(items),
                'processing_time': processing_time,
                'success': True
            })
            
            # Adjust batch size based on performance
            self._adjust_batch_size()
            
            logger.info(
                "Batch processing completed",
                batch_size=len(items),
                processing_time=processing_time,
                current_optimal_size=self.current_batch_size
            )
            
            return results
            
        except Exception as e:
            logger.error(
                "Batch processing failed",
                error=str(e),
                batch_size=len(items)
            )
            # Reduce batch size on error
            self.current_batch_size = max(
                self.min_batch_size,
                self.current_batch_size // 2
            )
            raise
    
    def _adjust_batch_size(self):
        """Adjust batch size based on performance history."""
        if len(self.performance_window) < 5:
            return
            
        # Calculate average processing time per item
        recent_performance = self.performance_window[-5:]
        avg_time_per_item = [
            p['processing_time'] / p['batch_size']
            for p in recent_performance
        ]
        
        if statistics.mean(avg_time_per_item) < 0.1:  # Less than 100ms per item
            # Increase batch size
            self.current_batch_size = min(
                self.max_batch_size,
                int(self.current_batch_size * 1.2)
            )
        elif statistics.mean(avg_time_per_item) > 0.2:  # More than 200ms per item
            # Decrease batch size
            self.current_batch_size = max(
                self.min_batch_size,
                int(self.current_batch_size * 0.8)
            )
```

### 4. Model Performance Monitoring

#### Scenario
Model inference times increasing over time.

#### Monitoring Configuration
```python
# 1. Model inference time
rate(model_inference_seconds_sum[5m]) / rate(model_inference_seconds_count[5m])

# 2. Model accuracy metrics
forecast_accuracy_mape

# 3. Model error rates
sum(rate(model_error_total[5m])) by (error_type)
```

#### Implementation Example
```python
from energy_forecast.core.utils.performance_monitoring import profiler
from energy_forecast.core.models.metrics import calculate_mape

class ModelPerformanceTracker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.performance_history = []
        self.accuracy_threshold = 0.15  # 15% MAPE threshold
        
    @profiler.profile_model(model_name="energy_forecast_v1")
    async def predict(self, features: dict) -> dict:
        """Make prediction with performance tracking."""
        start_time = time.time()
        
        try:
            # Model inference
            prediction = await self.model.predict(features)
            inference_time = time.time() - start_time
            
            # Record performance metrics
            self.performance_history.append({
                'timestamp': datetime.utcnow(),
                'inference_time': inference_time,
                'features': len(features)
            })
            
            # Calculate and log accuracy if actual value available
            if 'actual' in features:
                mape = calculate_mape(prediction['value'], features['actual'])
                logger.info(
                    "Model prediction completed",
                    model=self.model_name,
                    inference_time=inference_time,
                    mape=mape
                )
                
                # Alert if accuracy below threshold
                if mape > self.accuracy_threshold:
                    logger.warning(
                        "Model accuracy below threshold",
                        model=self.model_name,
                        mape=mape,
                        threshold=self.accuracy_threshold
                    )
            
            return prediction
            
        except Exception as e:
            logger.error(
                "Model prediction failed",
                error=str(e),
                model=self.model_name
            )
            raise
    
    def get_performance_stats(self) -> dict:
        """Calculate performance statistics."""
        recent_history = self.performance_history[-100:]  # Last 100 predictions
        
        return {
            'avg_inference_time': statistics.mean(
                [p['inference_time'] for p in recent_history]
            ),
            'p95_inference_time': statistics.quantiles(
                [p['inference_time'] for p in recent_history],
                n=20
            )[18],  # 95th percentile
            'prediction_count': len(recent_history),
            'timestamp': datetime.utcnow().isoformat()
        }
```

### 5. Database Performance Optimization

#### Scenario
Database query performance degradation.

#### Monitoring Metrics
```python
# 1. Query latency by type
rate(db_query_seconds_sum[5m]) / rate(db_query_seconds_count[5m])

# 2. Connection pool utilization
db_connection_pool_size

# 3. Query error rate
sum(rate(db_query_error_total[5m])) by (error_type)
```

#### Implementation Example
```python
from energy_forecast.core.utils.performance_monitoring import profiler
from energy_forecast.core.utils.database import DatabaseManager

class OptimizedDatabaseAccess:
    def __init__(self):
        self.db = DatabaseManager()
        self.slow_query_threshold = 1.0  # 1 second
        
    @profiler.profile_db("select")
    async def get_historical_data(
        self,
        city: str,
        start_date: str,
        end_date: str
    ) -> List[dict]:
        """Get historical data with performance monitoring."""
        query = """
        SELECT date, consumption
        FROM energy_consumption
        WHERE city = :city
          AND date BETWEEN :start_date AND :end_date
        """
        
        start_time = time.time()
        try:
            async with self.db.connection() as conn:
                result = await conn.fetch_all(
                    query,
                    {
                        'city': city,
                        'start_date': start_date,
                        'end_date': end_date
                    }
                )
                
            query_time = time.time() - start_time
            
            # Log slow queries
            if query_time > self.slow_query_threshold:
                logger.warning(
                    "Slow query detected",
                    query_time=query_time,
                    threshold=self.slow_query_threshold,
                    query=query,
                    params={
                        'city': city,
                        'start_date': start_date,
                        'end_date': end_date
                    }
                )
            
            return result
            
        except Exception as e:
            logger.error(
                "Database query failed",
                error=str(e),
                query=query,
                params={
                    'city': city,
                    'start_date': start_date,
                    'end_date': end_date
                }
            )
            raise
    
    @profiler.profile_db("health_check")
    async def check_database_health(self) -> dict:
        """Check database health metrics."""
        try:
            async with self.db.connection() as conn:
                # Check connection pool
                pool_stats = await conn.get_pool_stats()
                
                # Check query performance
                start_time = time.time()
                await conn.fetch_one("SELECT 1")
                ping_time = time.time() - start_time
                
                return {
                    'pool_size': pool_stats['size'],
                    'active_connections': pool_stats['active'],
                    'available_connections': pool_stats['available'],
                    'ping_time': ping_time,
                    'status': 'healthy' if ping_time < 0.1 else 'degraded'
                }
                
        except Exception as e:
            logger.error(
                "Database health check failed",
                error=str(e)
            )
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
```

## Alert Response Playbooks

### 1. High API Latency Alert
```yaml
alert: HighApiLatency
expr: rate(request_latency_seconds_sum[5m]) / rate(request_latency_seconds_count[5m]) > 2
for: 5m
labels:
  severity: warning
annotations:
  description: API latency is above 2 seconds for the last 5 minutes
  
response_steps:
  1. Check current load:
     - Monitor request rate
     - Check batch sizes
     - Review active connections
  
  2. Analyze cache performance:
     - Check hit ratio
     - Verify cache availability
     - Review cached data size
  
  3. Check database performance:
     - Monitor query latency
     - Review connection pool
     - Check for slow queries
  
  4. Scale if needed:
     - Increase cache size
     - Adjust batch sizes
     - Scale application instances
```

### 2. Low Cache Hit Ratio Alert
```yaml
alert: LowCacheHitRatio
expr: cache_hit_ratio < 0.5
for: 10m
labels:
  severity: warning
annotations:
  description: Cache hit ratio is below 50% for the last 10 minutes
  
response_steps:
  1. Analyze cache usage:
     - Review most accessed keys
     - Check TTL settings
     - Monitor memory usage
  
  2. Optimize caching strategy:
     - Adjust TTL values
     - Review cache key design
     - Consider prewarming
  
  3. Monitor impact:
     - Track API latency
     - Check database load
     - Monitor system resources
```

### 3. Database Connection Pool Alert
```yaml
alert: HighConnectionPoolUsage
expr: db_connection_pool_size > 80
for: 5m
labels:
  severity: critical
annotations:
  description: Database connection pool is above 80% capacity
  
response_steps:
  1. Immediate actions:
     - Check active connections
     - Monitor query duration
     - Review connection leaks
  
  2. Optimization:
     - Adjust pool size
     - Review query patterns
     - Check connection timeouts
  
  3. Long-term solutions:
     - Connection pooling strategy
     - Query optimization
     - Database scaling
```

## Performance Optimization Patterns

### 1. Batch Size Optimization
```python
def calculate_optimal_batch_size(
    current_performance: List[dict],
    min_size: int = 10,
    max_size: int = 100
) -> int:
    """Calculate optimal batch size based on performance history."""
    if not current_performance:
        return min_size
        
    # Calculate processing time per item for different batch sizes
    efficiency = {}
    for record in current_performance:
        batch_size = record['batch_size']
        time_per_item = record['processing_time'] / batch_size
        
        if batch_size not in efficiency:
            efficiency[batch_size] = []
        efficiency[batch_size].append(time_per_item)
    
    # Find most efficient batch size
    optimal_size = min_size
    best_efficiency = float('inf')
    
    for size, times in efficiency.items():
        avg_time = statistics.mean(times)
        if avg_time < best_efficiency:
            best_efficiency = avg_time
            optimal_size = size
    
    return optimal_size
```

### 2. Cache Warming Strategy
```python
async def warm_cache(
    cities: List[str],
    date_range: int = 7
) -> None:
    """Pre-warm cache with frequently accessed data."""
    start_date = datetime.now()
    dates = [
        (start_date + timedelta(days=i)).strftime('%Y-%m-%d')
        for i in range(date_range)
    ]
    
    for city in cities:
        for date in dates:
            cache_key = f"forecast:{city}:{date}"
            
            # Check if already cached
            if not await cache.exists(cache_key):
                try:
                    # Generate forecast
                    forecast = await model.predict(city, date)
                    
                    # Cache with appropriate TTL
                    await cache.set(
                        cache_key,
                        forecast,
                        expire=3600 * 24  # 24 hours
                    )
                    
                    logger.info(
                        "Cache warmed",
                        city=city,
                        date=date
                    )
                except Exception as e:
                    logger.error(
                        "Cache warming failed",
                        error=str(e),
                        city=city,
                        date=date
                    )
```

These examples provide practical implementations and monitoring strategies for common performance scenarios in the Energy Forecast platform.
