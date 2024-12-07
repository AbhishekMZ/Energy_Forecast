"""Model selection and ensemble methods"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import logging

from energy_forecast.core.models.baseline_models import BaselineModels
from energy_forecast.core.models.advanced_models import AdvancedModels
from energy_forecast.core.utils.error_handling import ProcessingError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelSelector:
    """Model selection and ensemble methods"""
    
    def __init__(self):
        self.baseline_models = BaselineModels()
        self.advanced_models = AdvancedModels()
        self.ensemble_weights = {}
        
    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Dict[str, Dict[str, float]]:
        """Train all available models and return their performance metrics"""
        try:
            metrics = {}
            
            # Train baseline models
            baseline_models = ['linear', 'ridge', 'lasso', 'random_forest']
            for model_name in baseline_models:
                self.baseline_models.train_model(model_name, X_train, y_train)
                y_pred = self.baseline_models.predict(model_name, X_train)
                metrics[model_name] = self.baseline_models.evaluate_model(
                    model_name, y_train, y_pred
                )
            
            # Train advanced models
            advanced_models = ['gbm', 'xgboost', 'svr']
            for model_name in advanced_models:
                self.advanced_models.train_model(model_name, X_train, y_train)
                y_pred = self.advanced_models.predict(model_name, X_train)
                metrics[model_name] = self.advanced_models.evaluate_model(
                    model_name, y_train, y_pred
                )
            
            return metrics
            
        except Exception as e:
            raise ProcessingError(
                "Error training models",
                {'original_error': str(e)}
            )
    
    def select_best_model(
        self,
        metrics: Dict[str, Dict[str, float]],
        metric: str = 'r2_score'
    ) -> str:
        """Select the best performing model based on specified metric"""
        try:
            model_scores = {
                model: scores[metric]
                for model, scores in metrics.items()
            }
            
            best_model = max(model_scores.items(), key=lambda x: x[1])[0]
            logger.info(f"Best model selected: {best_model}")
            
            return best_model
            
        except Exception as e:
            raise ProcessingError(
                "Error selecting best model",
                {'original_error': str(e)}
            )
    
    def create_weighted_ensemble(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        models: List[str],
        weights: Optional[List[float]] = None
    ) -> None:
        """Create a weighted ensemble of models"""
        try:
            if not weights:
                # Use equal weights if not specified
                weights = [1/len(models)] * len(models)
            
            if len(models) != len(weights):
                raise ValueError("Number of models must match number of weights")
            
            if not np.isclose(sum(weights), 1.0):
                raise ValueError("Weights must sum to 1")
            
            self.ensemble_weights = dict(zip(models, weights))
            logger.info("Created weighted ensemble")
            
        except Exception as e:
            raise ProcessingError(
                "Error creating ensemble",
                {'original_error': str(e)}
            )
    
    def predict_ensemble(
        self,
        X: pd.DataFrame
    ) -> pd.Series:
        """Make predictions using the weighted ensemble"""
        try:
            if not self.ensemble_weights:
                raise ValueError("Ensemble weights not set")
            
            predictions = pd.DataFrame()
            
            # Get predictions from all models in ensemble
            for model_name, weight in self.ensemble_weights.items():
                if model_name in ['linear', 'ridge', 'lasso', 'random_forest']:
                    pred = self.baseline_models.predict(model_name, X)
                else:
                    pred = self.advanced_models.predict(model_name, X)
                predictions[model_name] = pred
            
            # Calculate weighted average
            weighted_pred = pd.Series(0, index=X.index)
            for model_name, weight in self.ensemble_weights.items():
                weighted_pred += predictions[model_name] * weight
            
            return weighted_pred
            
        except Exception as e:
            raise ProcessingError(
                "Error making ensemble predictions",
                {'original_error': str(e)}
            )
    
    def optimize_ensemble_weights(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        models: List[str],
        cv: int = 5
    ) -> Dict[str, float]:
        """Optimize ensemble weights using cross-validation"""
        try:
            tscv = TimeSeriesSplit(n_splits=cv)
            
            # Initialize weights
            n_models = len(models)
            weights = np.ones(n_models) / n_models
            
            # Simple grid search for weights
            best_weights = weights
            best_score = float('-inf')
            
            for _ in range(100):  # Number of random trials
                # Generate random weights that sum to 1
                weights = np.random.dirichlet(np.ones(n_models))
                
                # Evaluate weights using cross-validation
                scores = []
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    # Train models
                    predictions = pd.DataFrame()
                    for i, model_name in enumerate(models):
                        if model_name in ['linear', 'ridge', 'lasso', 'random_forest']:
                            self.baseline_models.train_model(model_name, X_train, y_train)
                            pred = self.baseline_models.predict(model_name, X_val)
                        else:
                            self.advanced_models.train_model(model_name, X_train, y_train)
                            pred = self.advanced_models.predict(model_name, X_val)
                        predictions[model_name] = pred
                    
                    # Calculate weighted prediction
                    weighted_pred = np.zeros(len(y_val))
                    for i, model_name in enumerate(models):
                        weighted_pred += predictions[model_name] * weights[i]
                    
                    # Calculate score
                    score = r2_score(y_val, weighted_pred)
                    scores.append(score)
                
                avg_score = np.mean(scores)
                if avg_score > best_score:
                    best_score = avg_score
                    best_weights = weights
            
            # Set the best weights
            self.create_weighted_ensemble(X, y, models, best_weights.tolist())
            
            return dict(zip(models, best_weights))
            
        except Exception as e:
            raise ProcessingError(
                "Error optimizing ensemble weights",
                {'original_error': str(e)}
            )
