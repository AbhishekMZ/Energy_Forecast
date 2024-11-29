"""Optimized training pipeline with memory management, error handling, and performance optimization."""

import logging
from typing import Dict, List, Optional, Union, Tuple, Any
import pandas as pd
import numpy as np
from pathlib import Path

from utils.memory_management import MemoryManager, ChunkedDataLoader
from utils.error_handling import (
    ErrorHandler, ErrorMonitor, with_retry, error_context,
    DataError, ModelError, ConfigError, ResourceError
)
from utils.performance import (
    ResourceManager, ParallelProcessor, CacheManager,
    PerformanceMonitor
)

logger = logging.getLogger(__name__)

class OptimizedTrainingPipeline:
    """Training pipeline with advanced optimization and error handling."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize optimized training pipeline.
        
        Args:
            config: Configuration dictionary with training parameters
        """
        self.config = config
        
        # Initialize utility managers
        self.memory_manager = MemoryManager(threshold_gb=config.get('memory_threshold', 0.8))
        self.error_handler = ErrorHandler()
        self.resource_manager = ResourceManager()
        self.parallel_processor = ParallelProcessor()
        self.cache_manager = CacheManager()
        self.performance_monitor = PerformanceMonitor()
        self.error_monitor = ErrorMonitor()
        
        # Register error recovery strategies
        self._register_recovery_strategies()
    
    def _register_recovery_strategies(self):
        """Register error recovery strategies."""
        
        def handle_memory_error(error: ResourceError, context: Dict):
            """Handle memory-related errors."""
            logger.info("Attempting memory optimization...")
            if 'data' in context:
                # Try to optimize memory usage
                data = context['data']
                if isinstance(data, pd.DataFrame):
                    data = self.memory_manager.optimize_dataframe(data)
                return {'data': data, 'status': 'optimized'}
            raise error
        
        def handle_data_error(error: DataError, context: Dict):
            """Handle data-related errors."""
            logger.info("Attempting data recovery...")
            if 'data' in context:
                # Try to clean/validate data
                data = context['data']
                if isinstance(data, pd.DataFrame):
                    # Basic cleaning strategy
                    data = data.dropna()
                    return {'data': data, 'status': 'cleaned'}
            raise error
        
        # Register strategies
        self.error_handler.register_recovery_strategy(
            ResourceError, handle_memory_error
        )
        self.error_handler.register_recovery_strategy(
            DataError, handle_data_error
        )
    
    @performance_monitor.measure_time
    def load_and_prepare_data(self, data_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load and prepare data with optimization and error handling.
        
        Args:
            data_path: Path to data file
        
        Returns:
            Prepared DataFrame
        """
        with error_context("data_loading", self.error_handler):
            # Initialize chunked data loader
            loader = ChunkedDataLoader(data_path, chunk_size=self.config.get('chunk_size', 10000))
            
            def process_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
                # Optimize memory usage
                chunk = self.memory_manager.optimize_dataframe(chunk)
                
                # Apply preprocessing
                chunk = self._preprocess_chunk(chunk)
                
                return chunk
            
            # Process chunks in parallel
            processed_data = loader.process_chunks(process_chunk)
            
            return processed_data
    
    def _preprocess_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data chunk."""
        try:
            # Basic preprocessing
            chunk = chunk.copy()
            
            # Handle missing values
            chunk = chunk.fillna(chunk.mean(numeric_only=True))
            
            # Feature engineering
            if 'timestamp' in chunk.columns:
                chunk['hour'] = pd.to_datetime(chunk['timestamp']).dt.hour
                chunk['day_of_week'] = pd.to_datetime(chunk['timestamp']).dt.dayofweek
            
            return chunk
            
        except Exception as e:
            raise DataError(
                "Error preprocessing chunk",
                details={'error': str(e)}
            )
    
    @performance_monitor.measure_time
    @with_retry(max_retries=3)
    def train_model(self, model, X: pd.DataFrame, y: pd.Series) -> Any:
        """
        Train model with optimization and error handling.
        
        Args:
            model: Model instance
            X: Feature DataFrame
            y: Target series
        
        Returns:
            Trained model
        """
        with error_context("model_training", self.error_handler):
            # Monitor resources
            initial_resources = self.resource_manager.monitor_resources()
            
            try:
                # Train model
                model.fit(X, y)
                
                # Monitor final resources
                final_resources = self.resource_manager.monitor_resources()
                logger.info(
                    "Training completed. Memory usage: "
                    f"{final_resources['process_memory_gb'] - initial_resources['process_memory_gb']:.2f}GB"
                )
                
                return model
                
            except Exception as e:
                raise ModelError(
                    "Error training model",
                    details={
                        'error': str(e),
                        'model_type': type(model).__name__
                    }
                )
    
    @performance_monitor.measure_time
    def evaluate_model(self, model, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate model with optimization and error handling.
        
        Args:
            model: Trained model
            X: Feature DataFrame
            y: Target series
        
        Returns:
            Dictionary of evaluation metrics
        """
        with error_context("model_evaluation", self.error_handler):
            try:
                # Make predictions in chunks if data is large
                if len(X) > self.config.get('chunk_size', 10000):
                    predictions = []
                    with self.memory_manager.chunked_processing(X) as chunk:
                        chunk_pred = model.predict(chunk)
                        predictions.extend(chunk_pred)
                    y_pred = np.array(predictions)
                else:
                    y_pred = model.predict(X)
                
                # Calculate metrics
                metrics = self._calculate_metrics(y, y_pred)
                
                return metrics
                
            except Exception as e:
                raise ModelError(
                    "Error evaluating model",
                    details={
                        'error': str(e),
                        'model_type': type(model).__name__
                    }
                )
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
    
    @cache_manager.cache_result
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        return {
            'performance_stats': self.performance_monitor.get_performance_stats(),
            'error_stats': self.error_monitor.get_error_statistics(),
            'resource_usage': self.resource_manager.monitor_resources()
        }

# Example usage:
"""
# Configuration
config = {
    'memory_threshold': 0.8,
    'chunk_size': 10000,
    'n_workers': 4
}

# Initialize pipeline
pipeline = OptimizedTrainingPipeline(config)

# Load and prepare data
data = pipeline.load_and_prepare_data('data/train.csv')

# Split features and target
X = data.drop('target', axis=1)
y = data['target']

# Train model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
trained_model = pipeline.train_model(model, X, y)

# Evaluate model
metrics = pipeline.evaluate_model(trained_model, X, y)

# Get performance report
report = pipeline.get_performance_report()
"""
