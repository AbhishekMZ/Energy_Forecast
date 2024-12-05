from typing import Dict, List, Optional
from datetime import datetime, timedelta
import threading
import time
import logging
from collections import deque
from .state_machine import DataState, DataEvent, DataStateMachine
import psutil
import numpy as np
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class SystemMetrics:
    """
    System metrics collector using state machine concepts
    """
    def __init__(self, window_size: int = 3600):  # 1 hour window
        self.window_size = window_size
        self.metrics: Dict[str, deque] = {
            'data_processing_time': deque(maxlen=window_size),
            'validation_failures': deque(maxlen=window_size),
            'system_errors': deque(maxlen=window_size)
        }
        self.state_machine = DataStateMachine()
        self.logger = logging.getLogger(__name__)
        
    def record_metric(self, metric_name: str, value: float):
        """Record a metric with timestamp"""
        if metric_name in self.metrics:
            self.metrics[metric_name].append({
                'timestamp': datetime.now(),
                'value': value
            })
    
    def get_average(self, metric_name: str, minutes: int = 5) -> Optional[float]:
        """Get average of a metric over last n minutes"""
        if metric_name not in self.metrics:
            return None
            
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_values = [
            m['value'] for m in self.metrics[metric_name]
            if m['timestamp'] >= cutoff_time
        ]
        
        return sum(recent_values) / len(recent_values) if recent_values else None

class EnhancedMonitor:
    """Enhanced monitoring system with detailed metrics."""
    
    def __init__(self, history_size: int = 1000):
        """
        Initialize monitor.
        
        Args:
            history_size: Number of historical data points to keep
        """
        self.history_size = history_size
        self.history = deque(maxlen=history_size)
        self.model_metrics = {}
        self.error_stats = {}
        self.training_history = {}
        self.prediction_stats = {}
        
        # Start background monitoring
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.running:
            try:
                metrics = self._collect_metrics()
                self.history.append(metrics)
                time.sleep(1)  # Collect metrics every second
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
    
    def _collect_metrics(self) -> Dict[str, any]:
        """Collect current system metrics."""
        cpu = psutil.cpu_percent(interval=None, percpu=True)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        io = psutil.disk_io_counters()
        network = psutil.net_io_counters()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu': {
                'total': np.mean(cpu),
                'per_cpu': cpu,
                'frequency': psutil.cpu_freq().current if psutil.cpu_freq() else None,
                'load_avg': psutil.getloadavg()
            },
            'memory': {
                'total': memory.total,
                'available': memory.available,
                'used': memory.used,
                'percent': memory.percent,
                'swap_used': psutil.swap_memory().used if hasattr(psutil, 'swap_memory') else None
            },
            'disk': {
                'total': disk.total,
                'used': disk.used,
                'free': disk.free,
                'percent': disk.percent,
                'read_bytes': io.read_bytes,
                'write_bytes': io.write_bytes
            },
            'network': {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            },
            'processes': len(psutil.pids())
        }
    
    def record_model_metric(self, model_id: str, metric_name: str, value: float):
        """Record model-specific metric."""
        if model_id not in self.model_metrics:
            self.model_metrics[model_id] = {}
        if metric_name not in self.model_metrics[model_id]:
            self.model_metrics[model_id][metric_name] = []
        self.model_metrics[model_id][metric_name].append({
            'timestamp': datetime.now().isoformat(),
            'value': value
        })
    
    def record_error(self, error_type: str, error_msg: str):
        """Record error occurrence."""
        if error_type not in self.error_stats:
            self.error_stats[error_type] = {
                'count': 0,
                'first_seen': datetime.now().isoformat(),
                'examples': []
            }
        self.error_stats[error_type]['count'] += 1
        if len(self.error_stats[error_type]['examples']) < 5:
            self.error_stats[error_type]['examples'].append({
                'timestamp': datetime.now().isoformat(),
                'message': error_msg
            })
    
    def record_training_progress(self, model_id: str, metrics: Dict[str, float]):
        """Record training progress."""
        if model_id not in self.training_history:
            self.training_history[model_id] = []
        self.training_history[model_id].append({
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        })
    
    def record_prediction(self, model_id: str, prediction_time: float,
                         input_shape: tuple, confidence: Optional[float] = None):
        """Record prediction statistics."""
        if model_id not in self.prediction_stats:
            self.prediction_stats[model_id] = {
                'count': 0,
                'total_time': 0,
                'avg_time': 0,
                'min_time': float('inf'),
                'max_time': 0,
                'input_shapes': [],
                'confidences': []
            }
        
        stats = self.prediction_stats[model_id]
        stats['count'] += 1
        stats['total_time'] += prediction_time
        stats['avg_time'] = stats['total_time'] / stats['count']
        stats['min_time'] = min(stats['min_time'], prediction_time)
        stats['max_time'] = max(stats['max_time'], prediction_time)
        stats['input_shapes'].append(input_shape)
        if confidence is not None:
            stats['confidences'].append(confidence)
    
    def get_system_metrics(self, minutes: int = 5) -> Dict[str, any]:
        """Get system metrics for the last N minutes."""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        recent_history = [
            m for m in self.history
            if datetime.fromisoformat(m['timestamp']) > cutoff
        ]
        
        if not recent_history:
            return {}
        
        # Calculate statistics
        cpu_total = [m['cpu']['total'] for m in recent_history]
        memory_percent = [m['memory']['percent'] for m in recent_history]
        disk_percent = [m['disk']['percent'] for m in recent_history]
        
        return {
            'cpu': {
                'current': cpu_total[-1],
                'avg': np.mean(cpu_total),
                'max': np.max(cpu_total),
                'min': np.min(cpu_total)
            },
            'memory': {
                'current': memory_percent[-1],
                'avg': np.mean(memory_percent),
                'max': np.max(memory_percent),
                'min': np.min(memory_percent)
            },
            'disk': {
                'current': disk_percent[-1],
                'avg': np.mean(disk_percent),
                'max': np.max(disk_percent),
                'min': np.min(disk_percent)
            },
            'network': recent_history[-1]['network'],
            'processes': recent_history[-1]['processes']
        }
    
    def get_model_performance(self, model_id: str) -> Dict[str, any]:
        """Get comprehensive model performance metrics."""
        if model_id not in self.model_metrics:
            return {}
        
        metrics = self.model_metrics[model_id]
        training_hist = self.training_history.get(model_id, [])
        pred_stats = self.prediction_stats.get(model_id, {})
        
        return {
            'metrics': metrics,
            'training_progress': training_hist,
            'prediction_stats': pred_stats
        }
    
    def get_error_summary(self) -> Dict[str, any]:
        """Get error statistics summary."""
        return {
            'total_errors': sum(e['count'] for e in self.error_stats.values()),
            'error_types': len(self.error_stats),
            'most_common': sorted(
                self.error_stats.items(),
                key=lambda x: x[1]['count'],
                reverse=True
            )[:5],
            'details': self.error_stats
        }
    
    def save_metrics(self, output_dir: str):
        """Save all metrics to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save system metrics
        with open(output_path / 'system_metrics.json', 'w') as f:
            json.dump(list(self.history), f, indent=2)
        
        # Save model metrics
        with open(output_path / 'model_metrics.json', 'w') as f:
            json.dump(self.model_metrics, f, indent=2)
        
        # Save error stats
        with open(output_path / 'error_stats.json', 'w') as f:
            json.dump(self.error_stats, f, indent=2)
        
        # Save training history
        with open(output_path / 'training_history.json', 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Save prediction stats
        with open(output_path / 'prediction_stats.json', 'w') as f:
            json.dump(self.prediction_stats, f, indent=2)
    
    def __del__(self):
        """Cleanup monitoring thread."""
        self.running = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1)

class SystemMonitor:
    """
    System monitoring implementation using Theory of Computation concepts
    """
    def __init__(self):
        self.metrics = SystemMetrics()
        self.enhanced_monitor = EnhancedMonitor()
        self.alert_thresholds = {
            'data_processing_time': 5.0,  # seconds
            'validation_failures': 10,    # per minute
            'system_errors': 5           # per minute
        }
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.logger = logging.getLogger(__name__)
        
    def start_monitoring(self):
        """Start the monitoring thread"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop the monitoring thread"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                self._check_system_health()
                time.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
    
    def _check_system_health(self):
        """Check system health metrics"""
        for metric, threshold in self.alert_thresholds.items():
            avg_value = self.metrics.get_average(metric)
            if avg_value is not None and avg_value > threshold:
                self._trigger_alert(metric, avg_value, threshold)
    
    def _trigger_alert(self, metric: str, value: float, threshold: float):
        """Trigger system alert"""
        alert_message = (
            f"Alert: {metric} exceeded threshold. "
            f"Current value: {value:.2f}, Threshold: {threshold:.2f}"
        )
        self.logger.warning(alert_message)
        # Here you could add more alert mechanisms (email, SMS, etc.)
    
    def record_processing_time(self, duration: float):
        """Record data processing time"""
        self.metrics.record_metric('data_processing_time', duration)
    
    def record_validation_failure(self):
        """Record a validation failure"""
        self.metrics.record_metric('validation_failures', 1)
    
    def record_system_error(self):
        """Record a system error"""
        self.metrics.record_metric('system_errors', 1)
    
    def get_system_status(self) -> Dict:
        """Get current system status"""
        return {
            'data_processing_time': self.metrics.get_average('data_processing_time'),
            'validation_failures': self.metrics.get_average('validation_failures'),
            'system_errors': self.metrics.get_average('system_errors'),
            'is_healthy': self._is_system_healthy()
        }
    
    def _is_system_healthy(self) -> bool:
        """Check if system is healthy based on all metrics"""
        for metric, threshold in self.alert_thresholds.items():
            avg_value = self.metrics.get_average(metric)
            if avg_value is not None and avg_value > threshold:
                return False
        return True
