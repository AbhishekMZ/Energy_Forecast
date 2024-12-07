from typing import Dict, Any, List, Optional
import numpy as np

class ModelConfigurations:
    """Optimized model configurations for energy consumption forecasting"""
    
    @staticmethod
    def get_random_forest_config(data_size: int) -> Dict[str, Any]:
        """
        Get Random Forest configuration optimized for energy forecasting
        
        Parameters:
            data_size: Number of training samples
        """
        # Scale n_estimators based on data size
        n_estimators = min(max(int(data_size / 100), 100), 500)
        
        return {
            'n_estimators': n_estimators,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',  # Recommended for regression
            'bootstrap': True,
            'oob_score': True,  # Use out-of-bag score
            'n_jobs': -1,  # Use all CPU cores
            'random_state': 42,
            # Energy-specific parameters
            'criterion': 'squared_error',  # Better for continuous values
            'max_samples': 0.8,  # Reduce overfitting
            'warm_start': True,  # Enable incremental learning
        }
    
    @staticmethod
    def get_lightgbm_config(data_size: int, feature_count: int) -> Dict[str, Any]:
        """
        Get LightGBM configuration optimized for energy forecasting
        
        Parameters:
            data_size: Number of training samples
            feature_count: Number of features
        """
        return {
            'boosting_type': 'gbdt',
            'n_estimators': min(max(int(data_size / 100), 100), 500),
            'max_depth': 15,
            'num_leaves': 31,  # 2^(max_depth-1)
            'learning_rate': 0.05,
            'feature_fraction': 0.8,  # Feature subsampling
            'bagging_fraction': 0.8,  # Row subsampling
            'bagging_freq': 5,
            'min_data_in_leaf': max(20, int(data_size / 1000)),
            'lambda_l1': 0.1,  # L1 regularization
            'lambda_l2': 0.1,  # L2 regularization
            'min_gain_to_split': 0.0,
            'max_bin': 255,  # For memory efficiency
            'verbose': -1,
            'metric': ['rmse', 'mae'],
            'early_stopping_rounds': 50,
            # Energy-specific parameters
            'objective': 'regression',
            'first_metric_only': True,
            'boost_from_average': True,
            'force_col_wise': True  # Better for small datasets
        }
    
    @staticmethod
    def get_xgboost_config(data_size: int) -> Dict[str, Any]:
        """
        Get XGBoost configuration optimized for energy forecasting
        
        Parameters:
            data_size: Number of training samples
        """
        return {
            'n_estimators': min(max(int(data_size / 100), 100), 500),
            'max_depth': 7,
            'learning_rate': 0.1,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'colsample_bylevel': 0.8,
            'gamma': 0.1,  # Minimum loss reduction
            'reg_alpha': 0.1,  # L1 regularization
            'reg_lambda': 1.0,  # L2 regularization
            'scale_pos_weight': 1,
            'max_delta_step': 0,
            'random_state': 42,
            # Energy-specific parameters
            'objective': 'reg:squarederror',
            'tree_method': 'hist',  # Faster training
            'grow_policy': 'lossguide',
            'max_leaves': 31,
            'sampling_method': 'gradient_based',
            'eval_metric': ['rmse', 'mae']
        }
    
    @staticmethod
    def get_deep_learning_config(data_size: int,
                               feature_count: int,
                               sequence_length: Optional[int] = None) -> Dict[str, Any]:
        """
        Get Deep Learning configuration optimized for energy forecasting
        
        Parameters:
            data_size: Number of training samples
            feature_count: Number of features
            sequence_length: Length of time series sequence (if using sequential data)
        """
        # Scale architecture based on data complexity
        base_units = min(max(feature_count * 4, 32), 256)
        
        if sequence_length:
            # LSTM/GRU configuration for sequential data
            return {
                'architecture': 'sequential',
                'hidden_layers': [
                    {
                        'type': 'LSTM',
                        'units': base_units * 2,
                        'return_sequences': True
                    },
                    {
                        'type': 'LSTM',
                        'units': base_units,
                        'return_sequences': False
                    },
                    {
                        'type': 'Dense',
                        'units': base_units // 2
                    }
                ],
                'dropout_rate': 0.2,
                'recurrent_dropout': 0.1,
                'batch_normalization': True,
                'learning_rate': 0.001,
                'batch_size': min(max(data_size // 100, 32), 256),
                'epochs': 100,
                'early_stopping_patience': 10,
                'reduce_lr_patience': 5,
                'optimizer': {
                    'type': 'Adam',
                    'clipnorm': 1.0
                },
                'loss': 'huber',  # More robust to outliers
                'metrics': ['mae', 'mse']
            }
        else:
            # Dense neural network configuration
            return {
                'architecture': 'feedforward',
                'hidden_layers': [
                    base_units * 2,
                    base_units,
                    base_units // 2
                ],
                'activation': 'relu',
                'dropout_rate': 0.2,
                'batch_normalization': True,
                'learning_rate': 0.001,
                'batch_size': min(max(data_size // 100, 32), 256),
                'epochs': 100,
                'early_stopping_patience': 10,
                'reduce_lr_patience': 5,
                'optimizer': {
                    'type': 'Adam',
                    'clipnorm': 1.0
                },
                'loss': 'huber',
                'metrics': ['mae', 'mse']
            }
    
    @staticmethod
    def get_ensemble_config(models: List[str]) -> Dict[str, Any]:
        """
        Get Ensemble configuration optimized for energy forecasting
        
        Parameters:
            models: List of model names to include in ensemble
        """
        return {
            'models': models,
            'weights_method': 'dynamic',  # Dynamic weight calculation
            'stack_method': 'ridge',  # Use Ridge regression for stacking
            'cv_folds': 5,
            'refit_full': True,
            # Energy-specific parameters
            'weight_metric': 'rmse',  # Use RMSE for weight calculation
            'diversity_metric': 'correlation',  # Ensure model diversity
            'stack_params': {
                'alpha': 1.0,
                'fit_intercept': True,
                'normalize': True
            }
        }
    
    @staticmethod
    def get_time_series_config() -> Dict[str, Any]:
        """Get time series specific configuration for energy forecasting"""
        return {
            'seasonality': {
                'yearly': True,
                'weekly': True,
                'daily': True,
                'hourly': True
            },
            'decomposition': {
                'method': 'multiplicative',
                'period': 24  # For hourly data
            },
            'validation': {
                'method': 'rolling',
                'initial': '30D',
                'horizon': '7D',
                'step': '1D'
            },
            'feature_engineering': {
                'lag_features': [1, 24, 48, 168],  # Hour, day, 2 days, week
                'rolling_features': [6, 12, 24],  # Rolling means
                'ewm_features': [0.95, 0.97, 0.99],  # Exponential means
                'calendar_features': True
            }
        }
    
    @staticmethod
    def get_preprocessing_config() -> Dict[str, Any]:
        """Get preprocessing configuration for energy forecasting"""
        return {
            'outlier_detection': {
                'method': 'iqr',
                'threshold': 2.0,
                'rolling_window': 24
            },
            'missing_values': {
                'method': 'interpolate',
                'max_gap': 6,  # Maximum consecutive missing values to interpolate
                'default_method': 'time'  # Time-based interpolation
            },
            'scaling': {
                'method': 'robust',  # RobustScaler for handling outliers
                'quantile_range': (5, 95)
            },
            'feature_selection': {
                'method': 'recursive',
                'n_features_to_select': 'auto',
                'step': 0.1
            }
        }
    
    @staticmethod
    def adjust_for_data_size(config: Dict[str, Any],
                           data_size: int) -> Dict[str, Any]:
        """Adjust configuration based on dataset size"""
        if data_size < 1000:
            # Small dataset adjustments
            config['batch_size'] = min(32, data_size // 10)
            config['n_estimators'] = min(100, data_size // 5)
            config['early_stopping_rounds'] = 5
        elif data_size < 10000:
            # Medium dataset adjustments
            config['batch_size'] = 64
            config['n_estimators'] = 200
            config['early_stopping_rounds'] = 10
        else:
            # Large dataset adjustments
            config['batch_size'] = 128
            config['n_estimators'] = 500
            config['early_stopping_rounds'] = 20
        
        return config
