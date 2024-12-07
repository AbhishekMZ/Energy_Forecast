"""Advanced models for energy forecasting"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import GradientBoostingRegressor, XGBRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import logging

from energy_forecast.core.models.pipeline import MLPipeline
from energy_forecast.core.utils.error_handling import ProcessingError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedModels(MLPipeline):
    """Advanced models for energy forecasting"""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.available_models = {
            'gbm': self._train_gbm,
            'xgboost': self._train_xgboost,
            'svr': self._train_svr
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
    
    def _train_gbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        **kwargs
    ) -> None:
        """Train Gradient Boosting model"""
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        self.models['gbm'] = model
        self.feature_importance['gbm'] = pd.Series(
            model.feature_importances_,
            index=X_train.columns
        )
    
    def _train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        **kwargs
    ) -> None:
        """Train XGBoost model"""
        model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        self.models['xgboost'] = model
        self.feature_importance['xgboost'] = pd.Series(
            model.feature_importances_,
            index=X_train.columns
        )
    
    def _train_svr(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        kernel: str = 'rbf',
        C: float = 1.0,
        epsilon: float = 0.1,
        **kwargs
    ) -> None:
        """Train Support Vector Regression model"""
        model = SVR(
            kernel=kernel,
            C=C,
            epsilon=epsilon
        )
        model.fit(X_train, y_train)
        
        self.models['svr'] = model
        # SVR doesn't provide feature importance
        self.feature_importance['svr'] = None
    
    def hyperparameter_tuning(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: Dict,
        cv: int = 5
    ) -> Dict:
        """Perform hyperparameter tuning using GridSearchCV"""
        try:
            if model_name not in self.available_models:
                raise ValueError(f"Model {model_name} not available")
            
            # Get base model
            if model_name == 'gbm':
                base_model = GradientBoostingRegressor(random_state=42)
            elif model_name == 'xgboost':
                base_model = XGBRegressor(random_state=42)
            elif model_name == 'svr':
                base_model = SVR()
            else:
                raise ValueError(f"Hyperparameter tuning not implemented for {model_name}")
            
            # Perform grid search
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv,
                scoring='r2',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            # Store results
            tuning_results = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
            
            # Train model with best parameters
            self.train_model(
                model_name,
                X_train,
                y_train,
                **grid_search.best_params_
            )
            
            return tuning_results
            
        except Exception as e:
            raise ProcessingError(
                f"Error in hyperparameter tuning for {model_name}",
                {'original_error': str(e)}
            )
