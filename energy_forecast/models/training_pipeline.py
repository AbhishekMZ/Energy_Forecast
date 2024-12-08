"""Model retraining pipeline with feature engineering."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Any, List
import logging
from datetime import datetime, timedelta
import joblib
import os
from .versioning import ModelVersioning

logger = logging.getLogger(__name__)

class FeatureEngineering:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def create_time_features(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """Create time-based features from date column."""
        df = df.copy()
        df['hour'] = df[date_column].dt.hour
        df['day_of_week'] = df[date_column].dt.dayofweek
        df['month'] = df[date_column].dt.month
        df['is_weekend'] = df[date_column].dt.dayofweek.isin([5, 6]).astype(int)
        df['is_holiday'] = self._is_holiday(df[date_column])
        return df
        
    def create_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create weather-based features."""
        df = df.copy()
        df['temp_squared'] = df['temperature'] ** 2
        df['humidity_normalized'] = df['humidity'] / 100
        df['feels_like'] = self._calculate_feels_like(
            df['temperature'],
            df['humidity'],
            df['wind_speed']
        )
        return df
        
    def create_lag_features(
        self,
        df: pd.DataFrame,
        target_column: str,
        lag_periods: List[int]
    ) -> pd.DataFrame:
        """Create lagged features for the target variable."""
        df = df.copy()
        for lag in lag_periods:
            df[f'{target_column}_lag_{lag}'] = df.groupby('city')[target_column].shift(lag)
        return df
        
    def _calculate_feels_like(
        self,
        temp: pd.Series,
        humidity: pd.Series,
        wind_speed: pd.Series
    ) -> pd.Series:
        """Calculate feels like temperature using weather metrics."""
        return temp - 0.2 * (100 - humidity)/5 - wind_speed/10
        
    def _is_holiday(self, dates: pd.Series) -> pd.Series:
        """Determine if date is a holiday."""
        # Implement holiday detection logic here
        return pd.Series(0, index=dates.index)

class ModelTrainingPipeline:
    def __init__(self):
        self.feature_engineering = FeatureEngineering()
        self.model_versioning = ModelVersioning()
        self.model = None
        self.current_version = None
        
    def prepare_training_data(
        self,
        raw_data: pd.DataFrame,
        target_column: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for training with feature engineering."""
        try:
            # Create features
            data = self.feature_engineering.create_time_features(
                raw_data,
                'timestamp'
            )
            data = self.feature_engineering.create_weather_features(data)
            data = self.feature_engineering.create_lag_features(
                data,
                target_column,
                [1, 24, 168]  # 1 hour, 1 day, 1 week
            )
            
            # Handle missing values
            data = data.dropna()
            
            # Split features and target
            X = data.drop([target_column, 'timestamp'], axis=1)
            y = data[target_column]
            
            # Scale features
            X_scaled = pd.DataFrame(
                self.feature_engineering.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            
            return X_scaled, y
            
        except Exception as e:
            logger.error(f"Data preparation failed: {str(e)}")
            raise
            
    def train_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_params: Dict[str, Any]
    ) -> Dict[str, float]:
        """Train model and return performance metrics."""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            self.model.set_params(**model_params)
            self.model.fit(X_train, y_train)
            
            # Calculate metrics
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            metrics = {
                'train_score': train_score,
                'test_score': test_score,
                'train_size': len(X_train),
                'test_size': len(X_test)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise
            
    def save_model(
        self,
        metrics: Dict[str, float],
        params: Dict[str, Any]
    ) -> str:
        """Save trained model with versioning."""
        try:
            # Generate new version
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            version = f"v_{timestamp}"
            
            # Save model artifacts
            model_path = os.path.join('models', f'model_{version}.joblib')
            joblib.dump(self.model, model_path)
            
            # Log to model versioning system
            run_id = self.model_versioning.log_model_version(
                self.model,
                version,
                metrics,
                params
            )
            
            self.current_version = version
            return run_id
            
        except Exception as e:
            logger.error(f"Model saving failed: {str(e)}")
            raise
            
    def retrain_model(
        self,
        training_data: pd.DataFrame,
        model_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Complete retraining pipeline."""
        try:
            # Prepare data
            X, y = self.prepare_training_data(training_data, 'consumption_kwh')
            
            # Train model
            metrics = self.train_model(X, y, model_params)
            
            # Save model
            run_id = self.save_model(metrics, model_params)
            
            return {
                'version': self.current_version,
                'run_id': run_id,
                'metrics': metrics,
                'params': model_params
            }
            
        except Exception as e:
            logger.error(f"Retraining pipeline failed: {str(e)}")
            raise
