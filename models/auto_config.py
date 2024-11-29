from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
import warnings
from .model_configs import ModelConfigurations

class DataCharacteristics:
    """Analyze data characteristics for automatic model configuration"""
    
    def __init__(self, data: pd.DataFrame, target_col: str,
                 timestamp_col: Optional[str] = None):
        """
        Initialize data analyzer
        
        Parameters:
            data: Input DataFrame
            target_col: Name of target column
            timestamp_col: Name of timestamp column (if time series data)
        """
        self.data = data
        self.target_col = target_col
        self.timestamp_col = timestamp_col
        self.characteristics = {}
        
    def analyze_basic_stats(self) -> Dict[str, Any]:
        """Analyze basic statistical characteristics"""
        target = self.data[self.target_col]
        
        stats_dict = {
            'size': len(self.data),
            'n_features': len(self.data.columns) - (2 if self.timestamp_col else 1),
            'missing_ratio': self.data.isnull().mean().mean(),
            'target_stats': {
                'mean': target.mean(),
                'std': target.std(),
                'skew': target.skew(),
                'kurtosis': target.kurtosis(),
                'range': target.max() - target.min()
            }
        }
        
        # Detect outliers using IQR method
        Q1 = target.quantile(0.25)
        Q3 = target.quantile(0.75)
        IQR = Q3 - Q1
        stats_dict['outlier_ratio'] = (
            ((target < (Q1 - 1.5 * IQR)) | (target > (Q3 + 1.5 * IQR))).mean()
        )
        
        self.characteristics.update(stats_dict)
        return stats_dict
    
    def analyze_time_series_characteristics(self) -> Optional[Dict[str, Any]]:
        """Analyze time series specific characteristics"""
        if not self.timestamp_col:
            return None
            
        ts_data = self.data.set_index(self.timestamp_col)[self.target_col]
        
        # Resample to hourly frequency if needed
        if ts_data.index.freq is None:
            ts_data = ts_data.resample('H').mean()
        
        # Analyze seasonality and trend
        try:
            decomposition = seasonal_decompose(
                ts_data.interpolate(),
                period=24  # Assuming hourly data
            )
            
            trend_strength = np.abs(decomposition.trend).mean() / np.abs(ts_data).mean()
            seasonal_strength = np.abs(decomposition.seasonal).mean() / np.abs(ts_data).mean()
            
            # Test for stationarity
            adf_test = adfuller(ts_data.dropna())
            kpss_test = kpss(ts_data.dropna())
            
            ts_chars = {
                'trend_strength': trend_strength,
                'seasonal_strength': seasonal_strength,
                'is_stationary': adf_test[1] < 0.05 and kpss_test[1] >= 0.05,
                'autocorrelation': ts_data.autocorr(),
                'seasonality_detected': seasonal_strength > 0.1
            }
            
            self.characteristics.update({'time_series': ts_chars})
            return ts_chars
            
        except Exception as e:
            warnings.warn(f"Error in time series analysis: {str(e)}")
            return None
    
    def analyze_feature_relationships(self) -> Dict[str, Any]:
        """Analyze relationships between features and target"""
        feature_cols = [col for col in self.data.columns 
                       if col not in [self.target_col, self.timestamp_col]]
        
        relationships = {}
        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(self.data[col]):
                correlation = self.data[col].corr(self.data[self.target_col])
                relationships[col] = {
                    'correlation': correlation,
                    'importance_score': abs(correlation)
                }
        
        self.characteristics.update({'feature_relationships': relationships})
        return relationships
    
    def get_complexity_score(self) -> float:
        """Calculate overall data complexity score"""
        if not self.characteristics:
            self.analyze_basic_stats()
        
        # Factors that increase complexity
        complexity_factors = {
            'size': min(self.characteristics['size'] / 10000, 1),
            'features': min(self.characteristics['n_features'] / 50, 1),
            'missing_data': self.characteristics['missing_ratio'],
            'outliers': self.characteristics['outlier_ratio'],
            'nonlinearity': abs(self.characteristics['target_stats']['skew']) / 3
        }
        
        # Weight and combine factors
        weights = {
            'size': 0.2,
            'features': 0.2,
            'missing_data': 0.2,
            'outliers': 0.2,
            'nonlinearity': 0.2
        }
        
        complexity_score = sum(
            factor * weights[name]
            for name, factor in complexity_factors.items()
        )
        
        return min(max(complexity_score, 0), 1)


class AutoConfigTuner:
    """Automatic model configuration tuner based on data characteristics"""
    
    def __init__(self, data: pd.DataFrame, target_col: str,
                 timestamp_col: Optional[str] = None):
        """
        Initialize configuration tuner
        
        Parameters:
            data: Input DataFrame
            target_col: Name of target column
            timestamp_col: Name of timestamp column (if time series data)
        """
        self.data_analyzer = DataCharacteristics(data, target_col, timestamp_col)
        self.configs = ModelConfigurations()
        self.complexity_score = None
    
    def analyze_data(self) -> None:
        """Perform comprehensive data analysis"""
        self.data_analyzer.analyze_basic_stats()
        self.data_analyzer.analyze_feature_relationships()
        if self.data_analyzer.timestamp_col:
            self.data_analyzer.analyze_time_series_characteristics()
        self.complexity_score = self.data_analyzer.get_complexity_score()
    
    def _adjust_learning_rate(self, base_lr: float) -> float:
        """Adjust learning rate based on data characteristics"""
        if self.complexity_score > 0.7:
            return base_lr * 0.5  # Slower learning for complex data
        elif self.complexity_score < 0.3:
            return base_lr * 2.0  # Faster learning for simple data
        return base_lr
    
    def _adjust_model_capacity(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust model capacity based on complexity"""
        if 'n_estimators' in config:
            if self.complexity_score > 0.7:
                config['n_estimators'] = int(config['n_estimators'] * 1.5)
            elif self.complexity_score < 0.3:
                config['n_estimators'] = int(config['n_estimators'] * 0.7)
        
        if 'max_depth' in config:
            if self.complexity_score > 0.7:
                config['max_depth'] = min(int(config['max_depth'] * 1.3), 30)
            elif self.complexity_score < 0.3:
                config['max_depth'] = max(int(config['max_depth'] * 0.7), 3)
        
        return config
    
    def _adjust_regularization(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust regularization based on data characteristics"""
        chars = self.data_analyzer.characteristics
        
        # Increase regularization for noisy data
        if chars['outlier_ratio'] > 0.1 or chars['missing_ratio'] > 0.1:
            if 'reg_alpha' in config:
                config['reg_alpha'] *= 1.5
            if 'reg_lambda' in config:
                config['reg_lambda'] *= 1.5
            if 'dropout_rate' in config:
                config['dropout_rate'] = min(config['dropout_rate'] * 1.3, 0.5)
        
        return config
    
    def tune_random_forest(self) -> Dict[str, Any]:
        """Tune Random Forest configuration"""
        if not self.complexity_score:
            self.analyze_data()
            
        config = self.configs.get_random_forest_config(
            self.data_analyzer.characteristics['size']
        )
        
        # Adjust based on feature relationships
        feature_importances = [
            rel['importance_score']
            for rel in self.data_analyzer.characteristics['feature_relationships'].values()
        ]
        avg_importance = np.mean(feature_importances)
        
        if avg_importance < 0.3:
            config['max_features'] = 'sqrt'  # Try more feature combinations
        else:
            config['max_features'] = 0.8  # Use most features
        
        return self._adjust_model_capacity(config)
    
    def tune_lightgbm(self) -> Dict[str, Any]:
        """Tune LightGBM configuration"""
        if not self.complexity_score:
            self.analyze_data()
            
        config = self.configs.get_lightgbm_config(
            self.data_analyzer.characteristics['size'],
            self.data_analyzer.characteristics['n_features']
        )
        
        # Adjust learning rate
        config['learning_rate'] = self._adjust_learning_rate(config['learning_rate'])
        
        # Adjust for time series if applicable
        if 'time_series' in self.data_analyzer.characteristics:
            ts_chars = self.data_analyzer.characteristics['time_series']
            if ts_chars['seasonality_detected']:
                config['num_leaves'] = min(config['num_leaves'], 20)
            if not ts_chars['is_stationary']:
                config['learning_rate'] *= 0.8
        
        config = self._adjust_regularization(config)
        return self._adjust_model_capacity(config)
    
    def tune_xgboost(self) -> Dict[str, Any]:
        """Tune XGBoost configuration"""
        if not self.complexity_score:
            self.analyze_data()
            
        config = self.configs.get_xgboost_config(
            self.data_analyzer.characteristics['size']
        )
        
        # Adjust for data quality
        if self.data_analyzer.characteristics['missing_ratio'] > 0.1:
            config['gamma'] *= 1.5
        
        # Adjust for target distribution
        target_stats = self.data_analyzer.characteristics['target_stats']
        if abs(target_stats['skew']) > 1:
            config['max_delta_step'] = 1
        
        config = self._adjust_regularization(config)
        return self._adjust_model_capacity(config)
    
    def tune_deep_learning(self) -> Dict[str, Any]:
        """Tune Deep Learning configuration"""
        if not self.complexity_score:
            self.analyze_data()
            
        chars = self.data_analyzer.characteristics
        
        # Determine if we should use sequential model
        use_sequential = (
            'time_series' in chars and
            chars['time_series']['autocorrelation'] > 0.7
        )
        
        config = self.configs.get_deep_learning_config(
            chars['size'],
            chars['n_features'],
            sequence_length=24 if use_sequential else None
        )
        
        # Adjust architecture complexity
        if self.complexity_score > 0.7:
            config['hidden_layers'] = [units * 2 for units in config['hidden_layers']]
        elif self.complexity_score < 0.3:
            config['hidden_layers'] = [units // 2 for units in config['hidden_layers']]
        
        # Adjust regularization
        config['dropout_rate'] = min(
            config['dropout_rate'] * (1 + self.complexity_score),
            0.5
        )
        
        # Adjust learning process
        config['batch_size'] = int(
            config['batch_size'] * (2 - self.complexity_score)
        )
        config['learning_rate'] = self._adjust_learning_rate(config['learning_rate'])
        
        return config
    
    def tune_ensemble(self) -> Dict[str, Any]:
        """Tune Ensemble configuration"""
        if not self.complexity_score:
            self.analyze_data()
            
        # Select models based on data characteristics
        selected_models = []
        
        # Add tree-based models for nonlinear relationships
        if self.complexity_score > 0.5:
            selected_models.extend(['lightgbm', 'xgboost'])
        else:
            selected_models.append('random_forest')
        
        # Add neural network for complex patterns
        if (self.complexity_score > 0.6 and 
            self.data_analyzer.characteristics['size'] > 5000):
            selected_models.append('deep_learning')
        
        config = self.configs.get_ensemble_config(selected_models)
        
        # Adjust stacking parameters based on complexity
        if self.complexity_score > 0.7:
            config['stack_params']['alpha'] *= 1.5
        
        return config
    
    def get_optimal_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get optimal configurations for all models"""
        if not self.complexity_score:
            self.analyze_data()
            
        return {
            'random_forest': self.tune_random_forest(),
            'lightgbm': self.tune_lightgbm(),
            'xgboost': self.tune_xgboost(),
            'deep_learning': self.tune_deep_learning(),
            'ensemble': self.tune_ensemble()
        }
