from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
import optuna
import logging
from datetime import datetime
from .base_model import BaseModel
from .model_implementations import (
    RandomForestModel, LightGBMModel, XGBoostModel,
    DeepLearningModel, EnsembleModel
)
from ..utils.error_handling import ProcessingError
from ..database.operations import DatabaseOperations

class ModelTrainingPipeline:
    """Advanced model training pipeline with hyperparameter optimization"""
    
    def __init__(self, db_ops: DatabaseOperations):
        self.logger = logging.getLogger(__name__)
        self.db_ops = db_ops
        self.models = {}
        self.best_model = None
        self.feature_scaler = StandardScaler()
        
    def prepare_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for modeling"""
        try:
            # Sort by timestamp
            data = data.sort_values('timestamp')
            
            # Extract target variable
            target = data['consumption']
            features = data.drop(['consumption', 'timestamp'], axis=1)
            
            # Scale features
            scaled_features = pd.DataFrame(
                self.feature_scaler.fit_transform(features),
                columns=features.columns,
                index=features.index
            )
            
            return scaled_features, target
        except Exception as e:
            raise ProcessingError(
                "Error preparing data for modeling",
                {'original_error': str(e)}
            )
    
    def optimize_hyperparameters(self, model_class: type,
                               X_train: pd.DataFrame,
                               y_train: pd.Series,
                               n_trials: int = 100) -> Dict[str, Any]:
        """Optimize model hyperparameters using Optuna"""
        try:
            def objective(trial):
                # Define hyperparameter search space based on model type
                if model_class == RandomForestModel:
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                        'max_depth': trial.suggest_int('max_depth', 5, 30),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5)
                    }
                elif model_class in [LightGBMModel, XGBoostModel]:
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                        'max_depth': trial.suggest_int('max_depth', 3, 15),
                        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
                    }
                elif model_class == DeepLearningModel:
                    n_layers = trial.suggest_int('n_layers', 1, 5)
                    params = {
                        'hidden_layers': [
                            trial.suggest_int(f'units_l{i}', 32, 256)
                            for i in range(n_layers)
                        ],
                        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
                        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                        'batch_size': trial.suggest_int('batch_size', 16, 128)
                    }
                
                # Create and evaluate model
                model = model_class(params)
                
                # Use time series cross-validation
                tscv = TimeSeriesSplit(n_splits=5)
                scores = cross_val_score(
                    model.model, X_train, y_train,
                    cv=tscv, scoring='neg_mean_squared_error'
                )
                
                return -scores.mean()  # Return negative RMSE
            
            # Run optimization
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=n_trials)
            
            return study.best_params
            
        except Exception as e:
            raise ProcessingError(
                f"Error optimizing hyperparameters for {model_class.__name__}",
                {'original_error': str(e)}
            )
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                    optimize: bool = True) -> Dict[str, Dict[str, float]]:
        """Train multiple models and compare their performance"""
        try:
            results = {}
            
            # Define model classes to train
            model_classes = [
                RandomForestModel,
                LightGBMModel,
                XGBoostModel,
                DeepLearningModel
            ]
            
            # Train each model
            for model_class in model_classes:
                model_name = model_class.__name__
                self.logger.info(f"Training {model_name}...")
                
                # Optimize hyperparameters if requested
                if optimize:
                    params = self.optimize_hyperparameters(
                        model_class, X_train, y_train
                    )
                else:
                    params = None
                
                # Create and train model
                model = model_class(params)
                model.train(X_train, y_train)
                
                # Store model
                self.models[model_name] = model
                
                # Evaluate performance
                metrics = model.evaluate(X_train, y_train)
                results[model_name] = metrics
                
                # Save metrics to database
                self.db_ops.update_model_metrics(model_name, metrics)
            
            # Create and train ensemble model
            ensemble = EnsembleModel(list(self.models.values()))
            ensemble.train(X_train, y_train)
            self.models['Ensemble'] = ensemble
            
            # Evaluate ensemble
            ensemble_metrics = ensemble.evaluate(X_train, y_train)
            results['Ensemble'] = ensemble_metrics
            
            # Determine best model
            best_model_name = min(results.items(), key=lambda x: x[1]['rmse'])[0]
            self.best_model = self.models[best_model_name]
            
            return results
            
        except Exception as e:
            raise ProcessingError(
                "Error training models",
                {'original_error': str(e)}
            )
    
    def save_models(self, base_path: str) -> None:
        """Save all trained models"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            for model_name, model in self.models.items():
                path = f"{base_path}/{model_name}_{timestamp}.joblib"
                model.save_model(path)
                
        except Exception as e:
            raise ProcessingError(
                "Error saving models",
                {'base_path': base_path, 'original_error': str(e)}
            )
    
    def load_best_model(self, path: str) -> None:
        """Load the best performing model"""
        try:
            self.best_model = BaseModel.load_model(path)
        except Exception as e:
            raise ProcessingError(
                "Error loading best model",
                {'path': path, 'original_error': str(e)}
            )
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the best model"""
        try:
            if self.best_model is None:
                raise ProcessingError("No model has been trained or loaded")
            
            # Scale features
            X_scaled = pd.DataFrame(
                self.feature_scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
            
            return self.best_model.predict(X_scaled)
        except Exception as e:
            raise ProcessingError(
                "Error making predictions",
                {'original_error': str(e)}
            )
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """Get feature importance from the best model"""
        try:
            if self.best_model is None:
                raise ProcessingError("No model has been trained or loaded")
            
            return self.best_model.get_feature_importance()
        except Exception as e:
            raise ProcessingError(
                "Error getting feature importance",
                {'original_error': str(e)}
            )
