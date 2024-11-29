from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    LGBMRegressor, XGBRegressor
)
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LassoCV, RidgeCV
import tensorflow as tf
from .base_model import BaseModel
from ..utils.error_handling import ProcessingError

class RandomForestModel(BaseModel):
    """Random Forest implementation with advanced features"""
    
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        default_params = {
            'n_estimators': 200,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'n_jobs': -1,
            'random_state': 42
        }
        model_params = {**default_params, **(model_params or {})}
        super().__init__('RandomForest', model_params)
        self.model = RandomForestRegressor(**self.model_params)
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        try:
            self.model.fit(X_train, y_train)
            self.feature_importance = pd.Series(
                self.model.feature_importances_,
                index=X_train.columns
            ).sort_values(ascending=False)
        except Exception as e:
            raise ProcessingError(
                "Error training Random Forest model",
                {'original_error': str(e)}
            )
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        try:
            return self.model.predict(X)
        except Exception as e:
            raise ProcessingError(
                "Error making Random Forest predictions",
                {'original_error': str(e)}
            )

class LightGBMModel(BaseModel):
    """LightGBM implementation with advanced features"""
    
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        default_params = {
            'n_estimators': 200,
            'max_depth': 15,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'random_state': 42
        }
        model_params = {**default_params, **(model_params or {})}
        super().__init__('LightGBM', model_params)
        self.model = LGBMRegressor(**self.model_params)
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        try:
            self.model.fit(X_train, y_train)
            self.feature_importance = pd.Series(
                self.model.feature_importances_,
                index=X_train.columns
            ).sort_values(ascending=False)
        except Exception as e:
            raise ProcessingError(
                "Error training LightGBM model",
                {'original_error': str(e)}
            )
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        try:
            return self.model.predict(X)
        except Exception as e:
            raise ProcessingError(
                "Error making LightGBM predictions",
                {'original_error': str(e)}
            )

class XGBoostModel(BaseModel):
    """XGBoost implementation with advanced features"""
    
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        default_params = {
            'n_estimators': 200,
            'max_depth': 7,
            'learning_rate': 0.1,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        model_params = {**default_params, **(model_params or {})}
        super().__init__('XGBoost', model_params)
        self.model = XGBRegressor(**self.model_params)
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        try:
            self.model.fit(X_train, y_train)
            self.feature_importance = pd.Series(
                self.model.feature_importances_,
                index=X_train.columns
            ).sort_values(ascending=False)
        except Exception as e:
            raise ProcessingError(
                "Error training XGBoost model",
                {'original_error': str(e)}
            )
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        try:
            return self.model.predict(X)
        except Exception as e:
            raise ProcessingError(
                "Error making XGBoost predictions",
                {'original_error': str(e)}
            )

class DeepLearningModel(BaseModel):
    """Deep Learning implementation using TensorFlow"""
    
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        default_params = {
            'hidden_layers': [128, 64, 32],
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100
        }
        model_params = {**default_params, **(model_params or {})}
        super().__init__('DeepLearning', model_params)
        
    def _build_model(self, input_dim: int) -> None:
        """Build neural network architecture"""
        model = tf.keras.Sequential()
        
        # Input layer
        model.add(tf.keras.layers.Input(shape=(input_dim,)))
        
        # Hidden layers
        for units in self.model_params['hidden_layers']:
            model.add(tf.keras.layers.Dense(units, activation='relu'))
            model.add(tf.keras.layers.Dropout(self.model_params['dropout_rate']))
        
        # Output layer
        model.add(tf.keras.layers.Dense(1))
        
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.model_params['learning_rate']
        )
        
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        try:
            if self.model is None:
                self._build_model(X_train.shape[1])
            
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            history = self.model.fit(
                X_train, y_train,
                epochs=self.model_params['epochs'],
                batch_size=self.model_params['batch_size'],
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Store training history
            self.history = history.history
            
        except Exception as e:
            raise ProcessingError(
                "Error training Deep Learning model",
                {'original_error': str(e)}
            )
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        try:
            return self.model.predict(X, verbose=0).flatten()
        except Exception as e:
            raise ProcessingError(
                "Error making Deep Learning predictions",
                {'original_error': str(e)}
            )

class EnsembleModel(BaseModel):
    """Ensemble model combining multiple base models"""
    
    def __init__(self, base_models: List[BaseModel], weights: Optional[List[float]] = None):
        super().__init__('Ensemble')
        self.base_models = base_models
        self.weights = weights or [1/len(base_models)] * len(base_models)
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        try:
            for model in self.base_models:
                model.train(X_train, y_train)
        except Exception as e:
            raise ProcessingError(
                "Error training Ensemble model",
                {'original_error': str(e)}
            )
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        try:
            predictions = np.zeros((X.shape[0], len(self.base_models)))
            
            for i, model in enumerate(self.base_models):
                predictions[:, i] = model.predict(X) * self.weights[i]
                
            return predictions.sum(axis=1)
        except Exception as e:
            raise ProcessingError(
                "Error making Ensemble predictions",
                {'original_error': str(e)}
            )
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """Get aggregated feature importance from all models"""
        try:
            importance_dfs = []
            
            for model in self.base_models:
                imp = model.get_feature_importance()
                if imp is not None:
                    importance_dfs.append(imp)
            
            if importance_dfs:
                return pd.concat(importance_dfs, axis=1).mean(axis=1).sort_values(
                    ascending=False
                )
            return None
        except Exception:
            return None
