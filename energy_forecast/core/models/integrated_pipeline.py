"""Integrated pipeline combining all components of the energy forecasting system."""

import logging
from typing import Dict, List, Optional, Union, Tuple, Any
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from .auto_config import AutoConfigTuner
from .config_validation import ConfigValidator
from .config_visualization import ConfigVisualizer
from .model_evaluation import ModelEvaluator
from .model_implementations import (
    RandomForestModel, LightGBMModel, XGBoostModel,
    DeepLearningModel, EnsembleModel
)
from .optimized_training import OptimizedTrainingPipeline
from utils.memory_management import MemoryManager
from utils.error_handling import (
    ErrorHandler, error_context, DataError, ModelError
)
from utils.performance import (
    ResourceManager, PerformanceMonitor, CacheManager
)

logger = logging.getLogger(__name__)

class IntegratedPipeline:
    """Main pipeline integrating all system components."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize integrated pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path) if config_path else {}
        
        # Initialize components
        self.memory_manager = MemoryManager()
        self.error_handler = ErrorHandler()
        self.resource_manager = ResourceManager()
        self.performance_monitor = PerformanceMonitor()
        self.cache_manager = CacheManager()
        
        # Initialize ML components
        self.config_tuner = AutoConfigTuner()
        self.config_validator = ConfigValidator()
        self.config_visualizer = ConfigVisualizer()
        self.model_evaluator = ModelEvaluator()
        
        # Initialize optimized training pipeline
        self.training_pipeline = OptimizedTrainingPipeline(self.config)
        
        # Available models
        self.models = {
            'random_forest': RandomForestModel,
            'lightgbm': LightGBMModel,
            'xgboost': XGBoostModel,
            'deep_learning': DeepLearningModel,
            'ensemble': EnsembleModel
        }
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from file."""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    @performance_monitor.measure_time
    def run_pipeline(self, data_path: str, target_col: str,
                    model_type: str = 'ensemble',
                    optimize_config: bool = True,
                    validate_config: bool = True) -> Dict[str, Any]:
        """
        Run complete pipeline from data loading to model evaluation.
        
        Args:
            data_path: Path to input data
            target_col: Name of target column
            model_type: Type of model to use
            optimize_config: Whether to optimize configuration
            validate_config: Whether to validate configuration
        
        Returns:
            Dictionary with results and metrics
        """
        try:
            # Step 1: Load and prepare data
            logger.info("Loading and preparing data...")
            data = self.training_pipeline.load_and_prepare_data(data_path)
            
            # Split features and target
            X = data.drop(target_col, axis=1)
            y = data[target_col]
            
            # Step 2: Configuration optimization
            if optimize_config:
                logger.info("Optimizing configuration...")
                config = self.config_tuner.optimize_config(
                    data=data,
                    target_col=target_col
                )
            else:
                config = self.config.get('model_configs', {}).get(model_type, {})
            
            # Step 3: Configuration validation
            if validate_config:
                logger.info("Validating configuration...")
                self.config_validator.validate_config(config, model_type)
            
            # Step 4: Initialize model
            logger.info(f"Initializing {model_type} model...")
            model_class = self.models[model_type]
            model = model_class(**config)
            
            # Step 5: Train model
            logger.info("Training model...")
            trained_model = self.training_pipeline.train_model(model, X, y)
            
            # Step 6: Evaluate model
            logger.info("Evaluating model...")
            metrics = self.training_pipeline.evaluate_model(trained_model, X, y)
            
            # Step 7: Generate visualizations
            logger.info("Generating visualizations...")
            vis_data = self.config_visualizer.visualize_config(
                config=config,
                metrics=metrics,
                model_type=model_type
            )
            
            # Step 8: Compile results
            results = {
                'model': trained_model,
                'metrics': metrics,
                'config': config,
                'visualizations': vis_data,
                'performance_report': self.training_pipeline.get_performance_report()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def save_results(self, results: Dict[str, Any], output_dir: str):
        """Save pipeline results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics_path = output_path / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(results['metrics'], f, indent=4)
        
        # Save configuration
        config_path = output_path / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(results['config'], f, indent=4)
        
        # Save visualizations
        vis_path = output_path / 'visualizations'
        vis_path.mkdir(exist_ok=True)
        for name, fig in results['visualizations'].items():
            fig.write_html(str(vis_path / f'{name}.html'))
        
        # Save performance report
        report_path = output_path / 'performance_report.json'
        with open(report_path, 'w') as f:
            json.dump(results['performance_report'], f, indent=4)
        
        logger.info(f"Results saved to {output_path}")
    
    def load_model(self, model_path: str) -> Any:
        """Load trained model."""
        import joblib
        return joblib.load(model_path)
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return list(self.models.keys())
    
    def get_model_info(self, model_type: str) -> Dict[str, Any]:
        """Get information about specific model."""
        if model_type not in self.models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_class = self.models[model_type]
        return {
            'name': model_type,
            'description': model_class.__doc__,
            'parameters': model_class.get_parameter_info(),
            'requirements': model_class.get_requirements()
        }

# Example usage:
"""
# Initialize pipeline
pipeline = IntegratedPipeline('config.json')

# Run complete pipeline
results = pipeline.run_pipeline(
    data_path='data/train.csv',
    target_col='consumption',
    model_type='ensemble',
    optimize_config=True
)

# Save results
pipeline.save_results(results, 'output/experiment_1')

# Get model information
model_info = pipeline.get_model_info('random_forest')
"""
