"""Enhanced monitoring system with comprehensive metrics tracking."""

import psutil
import time
from datetime import datetime
from typing import Dict, List, Optional
import logging
from dataclasses import dataclass
import json
from prometheus_client import Counter, Histogram, Gauge
import traceback
from functools import wraps

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    name: str
    value: float
    metric_type: str
    tags: Dict[str, str]
    timestamp: datetime

class EnhancedMonitoring:
    def __init__(self, db_connection):
        self.db = db_connection
        
        # Prometheus metrics
        self.request_latency = Histogram(
            'request_latency_seconds',
            'Request latency in seconds',
            ['endpoint']
        )
        self.model_inference_time = Histogram(
            'model_inference_seconds',
            'Model inference time in seconds',
            ['model_version']
        )
        self.cache_hits = Counter(
            'cache_hits_total',
            'Number of cache hits',
            ['cache_type']
        )
        self.cache_misses = Counter(
            'cache_misses_total',
            'Number of cache misses',
            ['cache_type']
        )
        self.batch_size = Gauge(
            'batch_size_current',
            'Current batch size'
        )
        self.active_connections = Gauge(
            'db_connections_active',
            'Number of active database connections'
        )
        
    def monitor_performance(self, func):
        """Decorator for monitoring function performance."""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            error = None
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                error = e
                raise
            finally:
                duration = time.time() - start_time
                self.log_performance_metric(
                    name=f"{func.__name__}_duration",
                    value=duration,
                    metric_type="function_duration",
                    tags={
                        "function": func.__name__,
                        "error": str(error) if error else "none"
                    }
                )
        return wrapper
    
    def log_performance_metric(
        self,
        name: str,
        value: float,
        metric_type: str,
        tags: Optional[Dict[str, str]] = None
    ):
        """Log a performance metric to the database."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            metric_type=metric_type,
            tags=tags or {},
            timestamp=datetime.now()
        )
        
        with self.db.get_session() as session:
            session.execute(
                """
                INSERT INTO performance_metrics
                (metric_name, metric_value, metric_type, tags, timestamp)
                VALUES (:name, :value, :type, :tags, :timestamp)
                """,
                {
                    "name": metric.name,
                    "value": metric.value,
                    "type": metric.metric_type,
                    "tags": json.dumps(metric.tags),
                    "timestamp": metric.timestamp
                }
            )
    
    def log_error(
        self,
        error_type: str,
        error_message: str,
        severity: str,
        context: Optional[Dict] = None
    ):
        """Log an error with context."""
        with self.db.get_session() as session:
            session.execute(
                """
                INSERT INTO error_logs
                (error_type, error_message, stack_trace, context, severity)
                VALUES (:type, :message, :stack_trace, :context, :severity)
                """,
                {
                    "type": error_type,
                    "message": error_message,
                    "stack_trace": traceback.format_exc(),
                    "context": json.dumps(context) if context else None,
                    "severity": severity
                }
            )
    
    def log_model_performance(
        self,
        model_version_id: int,
        metric_name: str,
        metric_value: float,
        prediction_details: Optional[Dict] = None
    ):
        """Log model performance metrics."""
        with self.db.get_session() as session:
            session.execute(
                """
                INSERT INTO model_performance
                (model_version_id, metric_name, metric_value,
                prediction_id, actual_value, predicted_value)
                VALUES (:version_id, :metric_name, :metric_value,
                :pred_id, :actual, :predicted)
                """,
                {
                    "version_id": model_version_id,
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "pred_id": prediction_details.get("prediction_id") if prediction_details else None,
                    "actual": prediction_details.get("actual_value") if prediction_details else None,
                    "predicted": prediction_details.get("predicted_value") if prediction_details else None
                }
            )
    
    def update_system_health(self, component: str, status: str, details: Dict):
        """Update system health status."""
        with self.db.get_session() as session:
            session.execute(
                """
                INSERT INTO system_health
                (component, status, details, last_check)
                VALUES (:component, :status, :details, :last_check)
                """,
                {
                    "component": component,
                    "status": status,
                    "details": json.dumps(details),
                    "last_check": datetime.now()
                }
            )
    
    def log_batch_metrics(
        self,
        batch_id: str,
        batch_size: int,
        processing_time: float,
        success_count: int,
        error_count: int,
        start_time: datetime,
        end_time: datetime
    ):
        """Log batch processing metrics."""
        with self.db.get_session() as session:
            session.execute(
                """
                INSERT INTO batch_metrics
                (batch_id, batch_size, processing_time,
                success_count, error_count, start_time, end_time)
                VALUES (:batch_id, :size, :proc_time,
                :success, :error, :start, :end)
                """,
                {
                    "batch_id": batch_id,
                    "size": batch_size,
                    "proc_time": processing_time,
                    "success": success_count,
                    "error": error_count,
                    "start": start_time,
                    "end": end_time
                }
            )
    
    def get_system_metrics(self) -> Dict:
        """Get current system metrics."""
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "network_io": psutil.net_io_counters()._asdict(),
            "timestamp": datetime.now().isoformat()
        }
    
    def get_performance_summary(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> Dict:
        """Get performance summary for a time period."""
        with self.db.get_session() as session:
            metrics = session.execute(
                """
                SELECT metric_name, 
                       AVG(metric_value) as avg_value,
                       MIN(metric_value) as min_value,
                       MAX(metric_value) as max_value,
                       COUNT(*) as count
                FROM performance_metrics
                WHERE timestamp BETWEEN :start AND :end
                GROUP BY metric_name
                """,
                {"start": start_time, "end": end_time}
            ).fetchall()
            
            return {
                metric.metric_name: {
                    "average": metric.avg_value,
                    "min": metric.min_value,
                    "max": metric.max_value,
                    "count": metric.count
                }
                for metric in metrics
            }
    
    def get_error_summary(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> Dict:
        """Get error summary for a time period."""
        with self.db.get_session() as session:
            errors = session.execute(
                """
                SELECT error_type,
                       severity,
                       COUNT(*) as count
                FROM error_logs
                WHERE created_at BETWEEN :start AND :end
                GROUP BY error_type, severity
                """,
                {"start": start_time, "end": end_time}
            ).fetchall()
            
            return {
                error.error_type: {
                    "count": error.count,
                    "severity": error.severity
                }
                for error in errors
            }
    
    def get_model_performance_metrics(
        self,
        model_version_id: int,
        start_time: datetime,
        end_time: datetime
    ) -> Dict:
        """Get model performance metrics."""
        with self.db.get_session() as session:
            metrics = session.execute(
                """
                SELECT metric_name,
                       AVG(metric_value) as avg_value,
                       COUNT(*) as prediction_count
                FROM model_performance
                WHERE model_version_id = :version_id
                AND created_at BETWEEN :start AND :end
                GROUP BY metric_name
                """,
                {
                    "version_id": model_version_id,
                    "start": start_time,
                    "end": end_time
                }
            ).fetchall()
            
            return {
                metric.metric_name: {
                    "average": metric.avg_value,
                    "prediction_count": metric.prediction_count
                }
                for metric in metrics
            }
