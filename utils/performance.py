"""Performance optimization utilities for parallel processing and resource management."""

import multiprocessing as mp
from multiprocessing import Pool, Manager
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from typing import Callable, List, Any, Dict, Optional, Union, Tuple
import time
import logging
import functools
import numpy as np
import pandas as pd
from joblib import Memory
from pathlib import Path
import psutil
import os

logger = logging.getLogger(__name__)

class ResourceManager:
    """Manage and optimize system resources."""
    
    def __init__(self, cache_dir: str = '.cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.memory = Memory(str(self.cache_dir), verbose=0)
        self._process = psutil.Process(os.getpid())
    
    def get_optimal_workers(self) -> int:
        """Calculate optimal number of worker processes based on system resources."""
        cpu_count = mp.cpu_count()
        memory_available = psutil.virtual_memory().available / (1024 ** 3)  # GB
        
        # Estimate workers based on CPU and memory
        cpu_workers = max(1, cpu_count - 1)  # Leave one CPU free
        memory_workers = max(1, int(memory_available / 2))  # Assume 2GB per worker
        
        return min(cpu_workers, memory_workers)
    
    def monitor_resources(self) -> Dict[str, float]:
        """Monitor current resource usage."""
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available_gb': psutil.virtual_memory().available / (1024 ** 3),
            'process_memory_gb': self._process.memory_info().rss / (1024 ** 3)
        }

class ParallelProcessor:
    """Handle parallel processing with resource optimization."""
    
    def __init__(self, n_workers: Optional[int] = None):
        self.resource_manager = ResourceManager()
        self.n_workers = n_workers or self.resource_manager.get_optimal_workers()
    
    def process_parallel(self, func: Callable, items: List[Any], 
                        use_threads: bool = False, chunk_size: Optional[int] = None) -> List[Any]:
        """Process items in parallel using either processes or threads."""
        executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
        chunk_size = chunk_size or max(1, len(items) // (self.n_workers * 4))
        
        with executor_class(max_workers=self.n_workers) as executor:
            results = list(executor.map(func, items, chunksize=chunk_size))
        
        return results
    
    def parallel_dataframe(self, df: pd.DataFrame, func: Callable,
                          split_method: str = 'rows') -> pd.DataFrame:
        """Process DataFrame in parallel."""
        if split_method == 'rows':
            splits = np.array_split(df, self.n_workers)
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                results = list(executor.map(func, splits))
            return pd.concat(results)
        elif split_method == 'columns':
            df_split = {col: df[col] for col in df.columns}
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                results = {col: executor.submit(func, series) 
                          for col, series in df_split.items()}
                processed = {col: future.result() 
                           for col, future in results.items()}
            return pd.DataFrame(processed)
        else:
            raise ValueError(f"Unknown split method: {split_method}")

class CacheManager:
    """Manage caching of computation results."""
    
    def __init__(self, cache_dir: str = '.cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.memory = Memory(str(self.cache_dir), verbose=0)
    
    def cache_result(self, func: Callable) -> Callable:
        """Decorator to cache function results."""
        return self.memory.cache(func)
    
    def clear_cache(self, func_name: Optional[str] = None):
        """Clear cache for specific function or all cache."""
        if func_name:
            self.memory.forget(func_name)
        else:
            self.memory.clear()

class PerformanceMonitor:
    """Monitor and analyze performance metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, List[Dict]] = {}
    
    def measure_time(self, func: Callable) -> Callable:
        """Decorator to measure function execution time."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_resources = ResourceManager().monitor_resources()
            
            try:
                result = func(*args, **kwargs)
                success = True
            except Exception as e:
                success = False
                raise e
            finally:
                end_time = time.time()
                end_resources = ResourceManager().monitor_resources()
                
                metrics = {
                    'execution_time': end_time - start_time,
                    'success': success,
                    'timestamp': time.time(),
                    'resource_usage': {
                        'start': start_resources,
                        'end': end_resources,
                        'diff': {
                            k: end_resources[k] - start_resources[k]
                            for k in start_resources
                        }
                    }
                }
                
                if func.__name__ not in self.metrics:
                    self.metrics[func.__name__] = []
                self.metrics[func.__name__].append(metrics)
            
            return result
        return wrapper
    
    def get_performance_stats(self, func_name: Optional[str] = None) -> Dict:
        """Get performance statistics for function(s)."""
        if func_name:
            metrics = self.metrics.get(func_name, [])
        else:
            metrics = [m for metrics in self.metrics.values() for m in metrics]
        
        if not metrics:
            return {}
        
        execution_times = [m['execution_time'] for m in metrics]
        success_rate = sum(m['success'] for m in metrics) / len(metrics)
        
        return {
            'count': len(metrics),
            'success_rate': success_rate,
            'execution_time': {
                'mean': np.mean(execution_times),
                'std': np.std(execution_times),
                'min': np.min(execution_times),
                'max': np.max(execution_times)
            },
            'resource_usage': {
                'mean_cpu_diff': np.mean([
                    m['resource_usage']['diff']['cpu_percent']
                    for m in metrics
                ]),
                'mean_memory_diff': np.mean([
                    m['resource_usage']['diff']['memory_percent']
                    for m in metrics
                ])
            }
        }

# Example usage:
"""
# Initialize components
resource_manager = ResourceManager()
parallel_processor = ParallelProcessor()
cache_manager = CacheManager()
performance_monitor = PerformanceMonitor()

# Parallel processing
@performance_monitor.measure_time
def process_data_parallel(data):
    return parallel_processor.process_parallel(
        func=process_chunk,
        items=data_chunks
    )

# Caching
@cache_manager.cache_result
def expensive_computation(data):
    # Perform expensive computation
    return result

# Resource monitoring
@performance_monitor.measure_time
def resource_intensive_task():
    initial_resources = resource_manager.monitor_resources()
    # Perform task
    final_resources = resource_manager.monitor_resources()
    return final_resources['memory_available_gb'] - initial_resources['memory_available_gb']

# Get performance stats
stats = performance_monitor.get_performance_stats('process_data_parallel')
"""
