"""ML Pipeline for energy forecasting"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
from datetime import datetime
import joblib
from pathlib import Path

from energy_forecast.core.utils.data_quality import check_data_quality
from energy_forecast.core.utils.error_handling import ValidationError, ProcessingError
from energy_forecast.config.constants import MODEL_THRESHOLDS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLPipeline:
    """Machine Learning Pipeline for energy forecasting"""
    
    def __init__(self, config: Dict = None):
        self.config = config or MODEL_THRESHOLDS
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.metrics = {}
        
    def prepare_data(
        self,
        data: pd.DataFrame,
        target_col: str = 'total_demand',
        test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Prepare data for training"""
        try:
            # Validate data quality
            valid, errors = check_data_quality(data)
            if not valid:
                raise ValidationError("Data quality check failed", errors)
            
            # Create features
            X = self._create_features(data)
            y = data[target_col]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                shuffle=False  # Time series data should not be shuffled
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
            
            self.scalers['standard'] = scaler
            
            return X_train_scaled, X_test_scaled, y_train, y_test
            
        except Exception as e:
            raise ProcessingError(
                "Error preparing data",
                {'original_error': str(e)}
            )
    
    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features for model training"""
        df = data.copy()
        
        # Time-based features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Lag features
        df['demand_lag_1h'] = df['total_demand'].shift(1)
        df['demand_lag_24h'] = df['total_demand'].shift(24)
        df['demand_lag_168h'] = df['total_demand'].shift(168)  # 1 week
        
        # Rolling statistics
        df['demand_rolling_mean_24h'] = df['total_demand'].rolling(24).mean()
        df['demand_rolling_std_24h'] = df['total_demand'].rolling(24).std()
        
        # Weather features
        weather_cols = ['temperature', 'humidity', 'cloud_cover', 'solar_radiation']
        for col in weather_cols:
            if col in df.columns:
                df[f'{col}_lag_1h'] = df[col].shift(1)
                df[f'{col}_rolling_mean_24h'] = df[col].rolling(24).mean()
        
        # Drop rows with NaN values created by lag features
        df = df.dropna()
        
        # Drop the target variable and any unnecessary columns
        feature_cols = [col for col in df.columns if col != 'total_demand']
        
        return df[feature_cols]
    
    def evaluate_model(
        self,
        model_name: str,
        y_true: pd.Series,
        y_pred: pd.Series
    ) -> Dict[str, float]:
        """Evaluate model performance"""
        try:
            metrics = {
                'r2_score': r2_score(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred)
            }
            
            self.metrics[model_name] = metrics
            return metrics
            
        except Exception as e:
            raise ProcessingError(
                f"Error evaluating model {model_name}",
                {'original_error': str(e)}
            )
    
    def save_model(self, model_name: str, path: str) -> None:
        """Save model and associated objects"""
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            
            save_dict = {
                'model': self.models[model_name],
                'scaler': self.scalers.get('standard'),
                'feature_importance': self.feature_importance.get(model_name),
                'metrics': self.metrics.get(model_name),
                'config': self.config,
                'timestamp': datetime.now().isoformat()
            }
            
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(save_dict, path)
            logger.info(f"Model {model_name} saved to {path}")
            
        except Exception as e:
            raise ProcessingError(
                f"Error saving model {model_name}",
                {'path': path, 'original_error': str(e)}
            )
    
    def load_model(self, model_name: str, path: str) -> None:
        """Load model and associated objects"""
        try:
            save_dict = joblib.load(path)
            
            self.models[model_name] = save_dict['model']
            self.scalers['standard'] = save_dict['scaler']
            self.feature_importance[model_name] = save_dict['feature_importance']
            self.metrics[model_name] = save_dict['metrics']
            self.config = save_dict['config']
            
            logger.info(f"Model {model_name} loaded from {path}")
            
        except Exception as e:
            raise ProcessingError(
                f"Error loading model {model_name}",
                {'path': path, 'original_error': str(e)}
            )
