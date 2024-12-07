"""Baseline models for energy forecasting"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import logging

from energy_forecast.core.models.pipeline import MLPipeline
from energy_forecast.core.utils.error_handling import ProcessingError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaselineModels(MLPipeline):
    """Baseline models for energy forecasting"""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.available_models = {
            'linear': self._train_linear,
            'ridge': self._train_ridge,
            'lasso': self._train_lasso,
            'random_forest': self._train_random_forest
        }
    
    def train_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **kwargs
    ) -> None:
        """Train a specified model"""
        try:
            if model_name not in self.available_models:
                raise ValueError(f"Model {model_name} not available")
            
            logger.info(f"Training {model_name} model...")
            self.available_models[model_name](X_train, y_train, **kwargs)
            logger.info(f"Finished training {model_name} model")
            
        except Exception as e:
            raise ProcessingError(
                f"Error training model {model_name}",
                {'original_error': str(e)}
            )
    
    def predict(
        self,
        model_name: str,
        X: pd.DataFrame
    ) -> pd.Series:
        """Make predictions using a trained model"""
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not trained")
            
            predictions = self.models[model_name].predict(X)
            return pd.Series(predictions, index=X.index)
            
        except Exception as e:
            raise ProcessingError(
                f"Error making predictions with {model_name}",
                {'original_error': str(e)}
            )
    
    def _train_linear(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **kwargs
    ) -> None:
        """Train linear regression model"""
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        self.models['linear'] = model
        self.feature_importance['linear'] = pd.Series(
            model.coef_,
            index=X_train.columns
        )
    
    def _train_ridge(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        alpha: float = 1.0,
        **kwargs
    ) -> None:
        """Train ridge regression model"""
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        
        self.models['ridge'] = model
        self.feature_importance['ridge'] = pd.Series(
            model.coef_,
            index=X_train.columns
        )
    
    def _train_lasso(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        alpha: float = 1.0,
        **kwargs
    ) -> None:
        """Train lasso regression model"""
        model = Lasso(alpha=alpha)
        model.fit(X_train, y_train)
        
        self.models['lasso'] = model
        self.feature_importance['lasso'] = pd.Series(
            model.coef_,
            index=X_train.columns
        )
    
    def _train_random_forest(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        **kwargs
    ) -> None:
        """Train random forest model"""
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        self.models['random_forest'] = model
        self.feature_importance['random_forest'] = pd.Series(
            model.feature_importances_,
            index=X_train.columns
        )
    
    def cross_validate(
        self,
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5
    ) -> Dict[str, float]:
        """Perform cross-validation"""
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not trained")
            
            scores = cross_val_score(
                self.models[model_name],
                X, y,
                cv=cv,
                scoring='r2'
            )
            
            cv_results = {
                'mean_cv_score': scores.mean(),
                'std_cv_score': scores.std(),
                'cv_scores': scores.tolist()
            }
            
            return cv_results
            
        except Exception as e:
            raise ProcessingError(
                f"Error in cross-validation for {model_name}",
                {'original_error': str(e)}
            )
