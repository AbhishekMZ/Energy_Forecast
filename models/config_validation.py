from typing import Dict, Any, List, Tuple, Optional, Union
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from .auto_config import AutoConfigTuner
from .model_configs import ModelConfigurations

@dataclass
class ValidationResult:
    """Container for validation results"""
    model_name: str
    config: Dict[str, Any]
    metrics: Dict[str, float]
    fold_metrics: List[Dict[str, float]]
    validation_time: float
    
    @property
    def mean_rmse(self) -> float:
        return np.mean([m['rmse'] for m in self.fold_metrics])
    
    @property
    def std_rmse(self) -> float:
        return np.std([m['rmse'] for m in self.fold_metrics])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_name,
            'config': self.config,
            'mean_metrics': self.metrics,
            'fold_metrics': self.fold_metrics,
            'validation_time': self.validation_time,
            'rmse_stability': self.std_rmse / self.mean_rmse
        }


class ConfigurationValidator:
    """Validate model configurations using cross-validation"""
    
    def __init__(self, data: pd.DataFrame, target_col: str,
                 timestamp_col: Optional[str] = None):
        """
        Initialize validator
        
        Parameters:
            data: Input DataFrame
            target_col: Name of target column
            timestamp_col: Name of timestamp column (if time series data)
        """
        self.data = data
        self.target_col = target_col
        self.timestamp_col = timestamp_col
        self.logger = logging.getLogger(__name__)
        self.auto_tuner = AutoConfigTuner(data, target_col, timestamp_col)
        
    def _prepare_folds(self, n_splits: int = 5) -> Union[TimeSeriesSplit, KFold]:
        """Prepare cross-validation strategy"""
        if self.timestamp_col:
            # Time series cross-validation
            return TimeSeriesSplit(
                n_splits=n_splits,
                test_size=int(len(self.data) / (n_splits + 1))
            )
        else:
            # Standard k-fold cross-validation
            return KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    def _calculate_metrics(self, y_true: np.ndarray,
                         y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
    
    def _validate_single_config(self, model_class: type,
                              config: Dict[str, Any],
                              cv: Union[TimeSeriesSplit, KFold]) -> ValidationResult:
        """Validate a single model configuration"""
        start_time = datetime.now()
        fold_metrics = []
        
        feature_cols = [col for col in self.data.columns
                       if col not in [self.target_col, self.timestamp_col]]
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(self.data)):
            try:
                # Split data
                train_data = self.data.iloc[train_idx]
                val_data = self.data.iloc[val_idx]
                
                X_train = train_data[feature_cols]
                y_train = train_data[self.target_col]
                X_val = val_data[feature_cols]
                y_val = val_data[self.target_col]
                
                # Train model
                model = model_class(config)
                model.train(X_train, y_train)
                
                # Validate
                y_pred = model.predict(X_val)
                fold_metrics.append(self._calculate_metrics(y_val, y_pred))
                
            except Exception as e:
                self.logger.error(f"Error in fold {fold_idx}: {str(e)}")
                return None
        
        # Calculate aggregate metrics
        aggregate_metrics = {
            metric: np.mean([fold[metric] for fold in fold_metrics])
            for metric in fold_metrics[0].keys()
        }
        
        validation_time = (datetime.now() - start_time).total_seconds()
        
        return ValidationResult(
            model_name=model_class.__name__,
            config=config,
            metrics=aggregate_metrics,
            fold_metrics=fold_metrics,
            validation_time=validation_time
        )
    
    def validate_configurations(self,
                              n_splits: int = 5,
                              n_trials: int = 10) -> List[ValidationResult]:
        """Validate multiple configurations using cross-validation"""
        cv = self._prepare_folds(n_splits)
        results = []
        
        # Get base configurations
        self.auto_tuner.analyze_data()
        base_configs = self.auto_tuner.get_optimal_model_configs()
        
        def objective(trial, model_name: str,
                     base_config: Dict[str, Any]) -> float:
            """Objective function for hyperparameter optimization"""
            # Modify base configuration
            config = base_config.copy()
            
            if model_name in ['random_forest', 'lightgbm', 'xgboost']:
                config['n_estimators'] = trial.suggest_int(
                    'n_estimators',
                    int(base_config['n_estimators'] * 0.5),
                    int(base_config['n_estimators'] * 1.5)
                )
                config['max_depth'] = trial.suggest_int(
                    'max_depth',
                    max(3, int(base_config['max_depth'] * 0.5)),
                    int(base_config['max_depth'] * 1.5)
                )
            
            if model_name in ['lightgbm', 'xgboost']:
                config['learning_rate'] = trial.suggest_float(
                    'learning_rate',
                    base_config['learning_rate'] * 0.1,
                    base_config['learning_rate'] * 10,
                    log=True
                )
            
            if model_name == 'deep_learning':
                config['dropout_rate'] = trial.suggest_float(
                    'dropout_rate',
                    max(0.1, base_config['dropout_rate'] * 0.5),
                    min(0.5, base_config['dropout_rate'] * 2)
                )
                config['learning_rate'] = trial.suggest_float(
                    'learning_rate',
                    base_config['learning_rate'] * 0.1,
                    base_config['learning_rate'] * 10,
                    log=True
                )
            
            # Validate configuration
            result = self._validate_single_config(
                model_class=getattr(models, model_name),
                config=config,
                cv=cv
            )
            
            if result is None:
                return float('inf')
            
            return result.mean_rmse
        
        # Optimize each model's configuration
        for model_name, base_config in base_configs.items():
            try:
                study = optuna.create_study(direction='minimize')
                study.optimize(
                    lambda trial: objective(trial, model_name, base_config),
                    n_trials=n_trials
                )
                
                # Validate best configuration
                best_config = base_config.copy()
                best_config.update(study.best_params)
                
                result = self._validate_single_config(
                    model_class=getattr(models, model_name),
                    config=best_config,
                    cv=cv
                )
                
                if result is not None:
                    results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error optimizing {model_name}: {str(e)}")
        
        return results
    
    def analyze_validation_results(self,
                                 results: List[ValidationResult]) -> Dict[str, Any]:
        """Analyze validation results"""
        analysis = {
            'best_model': None,
            'model_rankings': [],
            'stability_analysis': {},
            'training_times': {},
            'detailed_metrics': {}
        }
        
        # Sort models by performance
        sorted_results = sorted(results, key=lambda x: x.mean_rmse)
        
        # Best model
        best_result = sorted_results[0]
        analysis['best_model'] = {
            'name': best_result.model_name,
            'config': best_result.config,
            'metrics': best_result.metrics
        }
        
        # Model rankings
        analysis['model_rankings'] = [
            {
                'name': r.model_name,
                'rmse': r.mean_rmse,
                'relative_performance': r.mean_rmse / best_result.mean_rmse
            }
            for r in sorted_results
        ]
        
        # Stability analysis
        for result in results:
            analysis['stability_analysis'][result.model_name] = {
                'rmse_cv': result.std_rmse / result.mean_rmse,  # Coefficient of variation
                'metric_stability': {
                    metric: np.std([fold[metric] for fold in result.fold_metrics])
                    for metric in result.fold_metrics[0].keys()
                }
            }
        
        # Training times
        analysis['training_times'] = {
            r.model_name: r.validation_time
            for r in results
        }
        
        # Detailed metrics
        analysis['detailed_metrics'] = {
            r.model_name: {
                'aggregate': r.metrics,
                'fold_metrics': r.fold_metrics
            }
            for r in results
        }
        
        return analysis
    
    def generate_validation_report(self, results: List[ValidationResult]) -> str:
        """Generate detailed validation report"""
        analysis = self.analyze_validation_results(results)
        
        report = [
            "# Model Configuration Validation Report",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Best Model Configuration",
            f"Model: {analysis['best_model']['name']}",
            "Metrics:",
            *[f"- {k}: {v:.4f}" for k, v in analysis['best_model']['metrics'].items()],
            "",
            "## Model Rankings",
            "| Model | RMSE | Relative Performance |",
            "|-------|------|---------------------|",
            *[f"| {r['name']} | {r['rmse']:.4f} | {r['relative_performance']:.2f}x |"
              for r in analysis['model_rankings']],
            "",
            "## Stability Analysis",
            "| Model | RMSE CV | MAE Stability | R² Stability |",
            "|-------|----------|---------------|--------------|",
            *[f"| {name} | {stats['rmse_cv']:.3f} | "
              f"{stats['metric_stability']['mae']:.3f} | "
              f"{stats['metric_stability']['r2']:.3f} |"
              for name, stats in analysis['stability_analysis'].items()],
            "",
            "## Training Times",
            "| Model | Time (seconds) |",
            "|-------|----------------|",
            *[f"| {name} | {time:.2f} |"
              for name, time in analysis['training_times'].items()]
        ]
        
        return "\n".join(report)
