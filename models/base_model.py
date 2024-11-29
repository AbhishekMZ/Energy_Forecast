from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import logging
from datetime import datetime
from ..utils.error_handling import ProcessingError

class BaseModel(ABC):
    """Abstract base class for all energy forecasting models"""
    
    def __init__(self, model_name: str, model_params: Optional[Dict[str, Any]] = None):
        self.model_name = model_name
        self.model_params = model_params or {}
        self.model = None
        self.feature_importance = None
        self.logger = logging.getLogger(__name__)
        
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        pass
    
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
                f"Error evaluating {self.model_name}",
                {'original_error': str(e)}
            )
    
    def save_model(self, path: str) -> None:
        """Save model to disk"""
        try:
            model_data = {
                'model': self.model,
                'model_name': self.model_name,
                'model_params': self.model_params,
                'feature_importance': self.feature_importance,
                'timestamp': datetime.now().isoformat()
            }
            joblib.dump(model_data, path)
        except Exception as e:
            raise ProcessingError(
                f"Error saving {self.model_name}",
                {'path': path, 'original_error': str(e)}
            )
    
    @classmethod
    def load_model(cls, path: str) -> 'BaseModel':
        """Load model from disk"""
        try:
            model_data = joblib.load(path)
            instance = cls(
                model_name=model_data['model_name'],
                model_params=model_data['model_params']
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
