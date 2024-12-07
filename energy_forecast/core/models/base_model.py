import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime
from abc import ABC, abstractmethod

# Update imports to use absolute imports from energy_forecast package
from energy_forecast.core.utils.data_quality import check_data_quality
from energy_forecast.core.utils.error_handling import ValidationError, ProcessingError
from energy_forecast.core.validation.basic_validator import BasicDataValidator
from energy_forecast.core.validation.statistical_validator import StatisticalValidator
from energy_forecast.config.constants import MODEL_THRESHOLDS
from energy_forecast.core.utils.data_pipeline import DataPipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """Base class for all energy forecast models"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.data_pipeline = DataPipeline()
        self.basic_validator = BasicDataValidator()
        self.statistical_validator = StatisticalValidator()
        self.model = None
        self.feature_importance = None
        
    @abstractmethod
    def train(self, data: pd.DataFrame) -> None:
        """Train the model"""
        pass
        
    @abstractmethod
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Make predictions"""
        pass
        
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate input data"""
        # Basic validation
        valid, basic_errors = self.basic_validator.run_all_validations(data)
        if not valid:
            return False, basic_errors
            
        # Statistical validation
        valid, stat_errors = self.statistical_validator.run_statistical_validations(data)
        return valid, stat_errors
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model performance"""
        try:
            predictions = self.predict(X_test)
            
            metrics = {
                'mse': mean_squared_error(y_test, predictions),
                'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
                'mae': mean_absolute_error(y_test, predictions),
                'r2': r2_score(y_test, predictions),
                'mape': np.mean(np.abs((y_test - predictions) / y_test)) * 100
            }
            
            return metrics
        except Exception as e:
            raise ProcessingError(
                f"Error evaluating model",
                {'original_error': str(e)}
            )
    
    def save_model(self, path: str) -> None:
        """Save model to disk"""
        try:
            model_data = {
                'model': self.model,
                'model_name': self.__class__.__name__,
                'model_params': self.config,
                'feature_importance': self.feature_importance,
                'timestamp': datetime.now().isoformat()
            }
            joblib.dump(model_data, path)
        except Exception as e:
            raise ProcessingError(
                f"Error saving model",
                {'path': path, 'original_error': str(e)}
            )
    
    @classmethod
    def load_model(cls, path: str) -> 'BaseModel':
        """Load model from disk"""
        try:
            model_data = joblib.load(path)
            instance = cls(
                config=model_data['model_params']
            )
            instance.model = model_data['model']
            instance.feature_importance = model_data['feature_importance']
            return instance
        except Exception as e:
            raise ProcessingError(
                "Error loading model",
                {'path': path, 'original_error': str(e)}
            )
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """Get feature importance if available"""
        return self.feature_importance if self.feature_importance is not None else None
