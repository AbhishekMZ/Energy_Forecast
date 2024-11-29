from typing import Dict, List, Optional
from datetime import datetime, timedelta
import threading
import time
import logging
from collections import deque
from .state_machine import DataState, DataEvent, DataStateMachine

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

class SystemMonitor:
    """
    System monitoring implementation using Theory of Computation concepts
    """
    def __init__(self):
        self.metrics = SystemMetrics()
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
