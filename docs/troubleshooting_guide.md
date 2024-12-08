# Troubleshooting Guide

## Overview

This guide provides solutions for common issues encountered in the Energy Forecast Platform, organized by system component.

## Quick Reference

| Issue Area | Common Symptoms | First Steps |
|------------|----------------|-------------|
| API Errors | 4xx/5xx responses | Check logs, verify API key |
| Model Predictions | Unexpected values | Verify input data quality |
| Data Pipeline | Missing/delayed data | Check data source connectivity |
| Performance | Slow response times | Monitor resource usage |
| Security | Authentication failures | Verify token validity |

## API Issues

### 1. Authentication Errors

```
HTTP 401: Unauthorized
```

**Common Causes:**
- Expired API token
- Invalid API key
- Incorrect authentication headers

**Solution:**
```python
# Check token validity
from core.auth import TokenValidator

validator = TokenValidator()
is_valid = validator.check_token(token)

if not is_valid:
    # Generate new token
    new_token = validator.generate_token(
        user_id=user.id,
        expiry=datetime.now() + timedelta(hours=24)
    )
```

### 2. Rate Limiting

```
HTTP 429: Too Many Requests
```

**Solution:**
```python
# Implement exponential backoff
async def retry_with_backoff(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await func()
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            wait_time = (2 ** attempt) * 1000  # ms
            await asyncio.sleep(wait_time / 1000)
```

## Model Issues

### 1. Prediction Anomalies

**Symptoms:**
- Unexpectedly high/low values
- Sudden changes in prediction patterns

**Diagnostic Steps:**
```python
# Check prediction statistics
def analyze_predictions(predictions: np.ndarray) -> Dict[str, float]:
    return {
        'mean': np.mean(predictions),
        'std': np.std(predictions),
        'min': np.min(predictions),
        'max': np.max(predictions),
        'anomaly_score': detect_anomalies(predictions)
    }

# Verify feature importance
def check_feature_importance(model, features: pd.DataFrame):
    importance = model.feature_importance()
    expected_importance = {
        'temperature': (0.2, 0.4),
        'time_of_day': (0.1, 0.3),
        'day_of_week': (0.05, 0.15)
    }
    
    for feature, (min_imp, max_imp) in expected_importance.items():
        actual_imp = importance[feature]
        if not min_imp <= actual_imp <= max_imp:
            logger.warning(
                f"Unexpected importance for {feature}: {actual_imp}"
            )
```

### 2. Model Performance Degradation

**Diagnostic Steps:**
```python
# Monitor prediction metrics
def monitor_model_performance(
    predictions: np.ndarray,
    actual: np.ndarray
):
    metrics = {
        'mape': mean_absolute_percentage_error(actual, predictions),
        'rmse': mean_squared_error(actual, predictions, squared=False),
        'r2': r2_score(actual, predictions)
    }
    
    # Check against thresholds
    thresholds = {
        'mape': 0.15,  # 15% MAPE
        'rmse': 100,   # 100 kWh
        'r2': 0.85     # R² score
    }
    
    for metric, value in metrics.items():
        if value > thresholds[metric]:
            alert_performance_degradation(metric, value)
```

## Data Pipeline Issues

### 1. Data Quality Problems

**Symptoms:**
- Missing values
- Inconsistent data types
- Out-of-range values

**Solution:**
```python
# Data quality checker
class DataQualityChecker:
    def check_quality(self, data: pd.DataFrame) -> List[str]:
        issues = []
        
        # Check completeness
        missing_pct = data.isna().sum() / len(data)
        for col, pct in missing_pct.items():
            if pct > 0.05:  # More than 5% missing
                issues.append(
                    f"High missing values in {col}: {pct:.2%}"
                )
        
        # Check value ranges
        for col, (min_val, max_val) in self.expected_ranges.items():
            out_of_range = data[
                ~data[col].between(min_val, max_val)
            ]
            if len(out_of_range) > 0:
                issues.append(
                    f"Out of range values in {col}: {len(out_of_range)} rows"
                )
        
        return issues
```

### 2. Pipeline Failures

**Diagnostic Steps:**
```python
# Pipeline health checker
class PipelineHealthChecker:
    def check_health(self) -> Dict[str, str]:
        status = {}
        
        # Check data sources
        for source in self.data_sources:
            try:
                source.test_connection()
                status[f"source_{source.name}"] = "healthy"
            except ConnectionError as e:
                status[f"source_{source.name}"] = f"unhealthy: {str(e)}"
        
        # Check processing stages
        for stage in self.processing_stages:
            try:
                stage.verify_state()
                status[f"stage_{stage.name}"] = "healthy"
            except Exception as e:
                status[f"stage_{stage.name}"] = f"error: {str(e)}"
        
        return status
```

## Performance Issues

### 1. Slow API Response

**Diagnostic Steps:**
```python
# Performance profiler
class APIProfiler:
    def profile_request(self, request_id: str):
        metrics = {
            'total_time': 0,
            'db_time': 0,
            'processing_time': 0,
            'serialization_time': 0
        }
        
        with Timer() as t:
            # Profile database queries
            with Timer() as db_timer:
                self.execute_db_queries()
            metrics['db_time'] = db_timer.duration
            
            # Profile data processing
            with Timer() as proc_timer:
                self.process_data()
            metrics['processing_time'] = proc_timer.duration
            
            # Profile serialization
            with Timer() as ser_timer:
                self.serialize_response()
            metrics['serialization_time'] = ser_timer.duration
        
        metrics['total_time'] = t.duration
        
        # Record metrics
        self.record_metrics(request_id, metrics)
```

### 2. High Resource Usage

**Solution:**
```python
# Resource monitor
class ResourceMonitor:
    def monitor_resources(self):
        # Monitor CPU usage
        cpu_usage = psutil.cpu_percent(interval=1)
        if cpu_usage > 80:
            self.alert_high_cpu(cpu_usage)
        
        # Monitor memory usage
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            self.alert_high_memory(memory.percent)
        
        # Monitor disk I/O
        disk_io = psutil.disk_io_counters()
        if disk_io.read_bytes > self.max_read_bytes:
            self.alert_high_disk_io('read', disk_io.read_bytes)
```

## Security Issues

### 1. Failed Authentication Attempts

**Solution:**
```python
# Security monitor
class SecurityMonitor:
    def monitor_auth_attempts(self):
        # Check recent failed attempts
        recent_failures = self.get_recent_failures(
            window_minutes=5
        )
        
        if len(recent_failures) > 10:
            # Potential brute force attack
            self.block_ip(recent_failures[0]['ip_address'])
            self.alert_security_team(
                'Potential brute force attack detected',
                recent_failures
            )
```

### 2. Suspicious Activity

**Detection:**
```python
# Activity analyzer
class ActivityAnalyzer:
    def analyze_activity(self, user_id: str):
        # Get user's normal pattern
        normal_pattern = self.get_user_pattern(user_id)
        
        # Check current activity
        current_activity = self.get_current_activity(user_id)
        
        # Compare with normal pattern
        anomalies = []
        for metric, value in current_activity.items():
            if abs(value - normal_pattern[metric]) > self.thresholds[metric]:
                anomalies.append({
                    'metric': metric,
                    'expected': normal_pattern[metric],
                    'actual': value
                })
        
        if anomalies:
            self.flag_suspicious_activity(user_id, anomalies)
```

## Recovery Procedures

### 1. Data Recovery

```python
# Data recovery manager
class DataRecoveryManager:
    async def recover_data(
        self,
        start_time: datetime,
        end_time: datetime
    ):
        # Check backup availability
        available_backups = self.list_backups(
            start_time,
            end_time
        )
        
        if not available_backups:
            raise NoBackupError(
                f"No backups found for period {start_time} to {end_time}"
            )
        
        # Restore from most recent backup
        latest_backup = available_backups[0]
        await self.restore_from_backup(latest_backup)
        
        # Verify restoration
        if not self.verify_data_integrity():
            raise RecoveryError("Data integrity check failed")
```

### 2. Service Recovery

```python
# Service recovery manager
class ServiceRecoveryManager:
    async def recover_service(self, service_name: str):
        # Stop affected service
        await self.stop_service(service_name)
        
        # Check dependencies
        deps_status = await self.check_dependencies(service_name)
        if not deps_status['healthy']:
            await self.recover_dependencies(deps_status['unhealthy'])
        
        # Reset service state
        await self.reset_service_state(service_name)
        
        # Restart service
        await self.start_service(service_name)
        
        # Verify recovery
        if not await self.verify_service_health(service_name):
            raise RecoveryError(f"Failed to recover {service_name}")
```

## Monitoring and Alerts

### 1. Alert Configuration

```python
# Alert configuration
ALERT_CONFIG = {
    'high_priority': {
        'channels': ['email', 'slack', 'pager'],
        'retry_interval': 300,  # 5 minutes
        'max_retries': 3
    },
    'medium_priority': {
        'channels': ['email', 'slack'],
        'retry_interval': 900,  # 15 minutes
        'max_retries': 2
    },
    'low_priority': {
        'channels': ['slack'],
        'retry_interval': 3600,  # 1 hour
        'max_retries': 1
    }
}
```

### 2. Health Checks

```python
# Health check manager
class HealthCheckManager:
    async def run_health_checks(self):
        results = {}
        
        # Check API health
        results['api'] = await self.check_api_health()
        
        # Check database health
        results['database'] = await self.check_database_health()
        
        # Check model health
        results['model'] = await self.check_model_health()
        
        # Check pipeline health
        results['pipeline'] = await self.check_pipeline_health()
        
        return results
```

## Common Error Messages

### API Errors

| Error Code | Message | Solution |
|------------|---------|----------|
| ERR_AUTH_001 | "Invalid API key" | Verify API key in configuration |
| ERR_AUTH_002 | "Token expired" | Refresh authentication token |
| ERR_RATE_001 | "Rate limit exceeded" | Implement request throttling |
| ERR_DATA_001 | "Invalid input format" | Check input data schema |

### Model Errors

| Error Code | Message | Solution |
|------------|---------|----------|
| ERR_MODEL_001 | "Invalid feature values" | Verify input data ranges |
| ERR_MODEL_002 | "Prediction failed" | Check model state and logs |
| ERR_MODEL_003 | "Model not found" | Verify model path and version |

## Contact Information

### Support Channels

- **Technical Support**: support@energyforecast.com
- **Security Issues**: security@energyforecast.com
- **Emergency Contact**: +1-XXX-XXX-XXXX

### Escalation Path

1. L1 Support Team
2. System Engineers
3. DevOps Team
4. Platform Architects
5. CTO Office

## Additional Resources

- [System Architecture Documentation](./system_architecture.md)
- [API Documentation](./api_reference.md)
- [Monitoring Guide](./monitoring_guide.md)
- [Security Guide](./security_guide.md)
- [Deployment Guide](./deployment_guide.md)
