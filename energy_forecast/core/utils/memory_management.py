"""Memory management utilities for handling large datasets and efficient resource usage."""

import gc
import os
import psutil
import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, List, Tuple
from functools import wraps
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class MemoryManager:
    """Memory management utilities for large data processing."""
    
    def __init__(self, threshold_gb: float = 0.8):
        """
        Initialize memory manager.
        
        Args:
            threshold_gb: Memory usage threshold in GB before optimization
        """
        self.threshold_gb = threshold_gb
        self._process = psutil.Process(os.getpid())
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        memory_info = self._process.memory_info()
        return {
            'rss_gb': memory_info.rss / (1024 ** 3),  # Resident Set Size
            'vms_gb': memory_info.vms / (1024 ** 3),  # Virtual Memory Size
            'percent': self._process.memory_percent(),
            'available_gb': psutil.virtual_memory().available / (1024 ** 3)
        }
    
    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage by downcasting types."""
        
        def _downcast_numeric(series: pd.Series) -> pd.Series:
            if pd.api.types.is_integer_dtype(series):
                min_val, max_val = series.min(), series.max()
                if min_val >= 0:
                    if max_val <= 255:
                        return series.astype(np.uint8)
                    elif max_val <= 65535:
                        return series.astype(np.uint16)
                    elif max_val <= 4294967295:
                        return series.astype(np.uint32)
                    return series.astype(np.uint64)
                else:
                    if min_val >= -128 and max_val <= 127:
                        return series.astype(np.int8)
                    elif min_val >= -32768 and max_val <= 32767:
                        return series.astype(np.int16)
                    elif min_val >= -2147483648 and max_val <= 2147483647:
                        return series.astype(np.int32)
                    return series.astype(np.int64)
            elif pd.api.types.is_float_dtype(series):
                return series.astype(np.float32)
            return series
        
        original_memory = df.memory_usage(deep=True).sum() / (1024 ** 2)
        
        # Optimize numeric columns
        for col in df.select_dtypes(include=['int64', 'float64']).columns:
            df[col] = _downcast_numeric(df[col])
        
        # Optimize categorical columns
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) < 0.5:  # If less than 50% unique values
                df[col] = df[col].astype('category')
        
        new_memory = df.memory_usage(deep=True).sum() / (1024 ** 2)
        logger.info(f"Memory usage reduced from {original_memory:.2f}MB to {new_memory:.2f}MB")
        
        return df
    
    @contextmanager
    def chunked_processing(self, df: pd.DataFrame, chunk_size: int = 10000):
        """Process large DataFrames in chunks to manage memory."""
        try:
            total_rows = len(df)
            for start_idx in range(0, total_rows, chunk_size):
                end_idx = min(start_idx + chunk_size, total_rows)
                chunk = df.iloc[start_idx:end_idx].copy()
                yield chunk
                del chunk
                gc.collect()
        finally:
            gc.collect()
    
    def monitor_memory(self, threshold_callback=None):
        """Decorator to monitor memory usage during function execution."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                initial_usage = self.get_memory_usage()
                logger.info(f"Initial memory usage: {initial_usage['rss_gb']:.2f}GB")
                
                try:
                    result = func(*args, **kwargs)
                    
                    final_usage = self.get_memory_usage()
                    logger.info(f"Final memory usage: {final_usage['rss_gb']:.2f}GB")
                    
                    if final_usage['rss_gb'] > self.threshold_gb and threshold_callback:
                        threshold_callback(final_usage)
                    
                    return result
                finally:
                    gc.collect()
            return wrapper
        return decorator
    
    def clear_memory(self):
        """Force garbage collection and memory cleanup."""
        gc.collect()
        if hasattr(torch, 'cuda'):
            import torch
            torch.cuda.empty_cache()
    
    @contextmanager
    def temporary_clone(self, df: pd.DataFrame):
        """Create a temporary copy of data and ensure cleanup."""
        temp_df = df.copy()
        try:
            yield temp_df
        finally:
            del temp_df
            gc.collect()
    
    def estimate_memory_usage(self, df: pd.DataFrame) -> Dict[str, float]:
        """Estimate memory usage for DataFrame operations."""
        memory_usage = df.memory_usage(deep=True).sum() / (1024 ** 3)  # GB
        estimated_peak = memory_usage * 3  # Estimate peak usage during operations
        
        return {
            'current_gb': memory_usage,
            'estimated_peak_gb': estimated_peak,
            'available_gb': psutil.virtual_memory().available / (1024 ** 3)
        }

class ChunkedDataLoader:
    """Load and process large datasets in chunks."""
    
    def __init__(self, filepath: str, chunk_size: int = 10000):
        self.filepath = filepath
        self.chunk_size = chunk_size
        self.memory_manager = MemoryManager()
    
    def process_chunks(self, processing_func):
        """Process data chunks with custom function."""
        chunks = pd.read_csv(self.filepath, chunksize=self.chunk_size)
        results = []
        
        for chunk in chunks:
            # Optimize memory usage for chunk
            chunk = self.memory_manager.optimize_dataframe(chunk)
            
            # Process chunk
            result = processing_func(chunk)
            results.append(result)
            
            # Clean up
            del chunk
            gc.collect()
        
        return pd.concat(results) if isinstance(results[0], pd.DataFrame) else results

def optimize_numeric_dtypes(arr: np.ndarray) -> np.ndarray:
    """Optimize numeric array dtypes to minimize memory usage."""
    if arr.dtype in [np.float64, np.float32]:
        # Check if we can convert to integer
        if np.array_equal(arr, arr.astype(int)):
            arr = arr.astype(int)
    
    if arr.dtype in [np.int64, np.int32]:
        min_val, max_val = arr.min(), arr.max()
        if min_val >= 0:
            if max_val <= 255:
                return arr.astype(np.uint8)
            elif max_val <= 65535:
                return arr.astype(np.uint16)
        else:
            if min_val >= -128 and max_val <= 127:
                return arr.astype(np.int8)
            elif min_val >= -32768 and max_val <= 32767:
                return arr.astype(np.int16)
    
    return arr

# Usage example:
"""
# Initialize memory manager
memory_manager = MemoryManager(threshold_gb=0.8)

# Monitor memory usage in functions
@memory_manager.monitor_memory()
def process_large_dataset(data):
    # Process data in chunks
    with memory_manager.chunked_processing(data, chunk_size=10000) as chunk:
        # Process chunk
        result = perform_operations(chunk)
    return result

# Load large datasets
loader = ChunkedDataLoader('large_dataset.csv', chunk_size=10000)
results = loader.process_chunks(lambda chunk: process_chunk(chunk))
"""
