from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
from sklearn.model_selection import TimeSeriesSplit
import logging
from datetime import datetime, timedelta
from ..utils.error_handling import ProcessingError

class ModelEvaluator:
    """Advanced model evaluation and comparison framework"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.evaluation_results = {}
        
    def evaluate_predictions(self, y_true: pd.Series,
                           y_pred: np.ndarray,
                           model_name: str) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        try:
            metrics = {
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'mape': mean_absolute_percentage_error(y_true, y_pred) * 100,
                'r2': r2_score(y_true, y_pred)
            }
            
            # Calculate additional metrics
            metrics.update({
                'median_ae': np.median(np.abs(y_true - y_pred)),
                'max_error': np.max(np.abs(y_true - y_pred)),
                'std_error': np.std(y_true - y_pred)
            })
            
            self.evaluation_results[model_name] = metrics
            return metrics
            
        except Exception as e:
            raise ProcessingError(
                "Error calculating evaluation metrics",
                {'model': model_name, 'original_error': str(e)}
            )
    
    def compare_models(self, results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Compare multiple models' performance"""
        try:
            comparison_df = pd.DataFrame(results).T
            comparison_df = comparison_df.round(4)
            comparison_df = comparison_df.sort_values('rmse')
            
            return comparison_df
            
        except Exception as e:
            raise ProcessingError(
                "Error comparing models",
                {'original_error': str(e)}
            )
    
    def plot_predictions(self, y_true: pd.Series,
                        predictions: Dict[str, np.ndarray],
                        title: str = "Model Predictions Comparison") -> plt.Figure:
        """Plot actual vs predicted values for multiple models"""
        try:
            plt.figure(figsize=(12, 6))
            
            # Plot actual values
            plt.plot(y_true.index, y_true.values, label='Actual', linewidth=2)
            
            # Plot predictions for each model
            for model_name, y_pred in predictions.items():
                plt.plot(y_true.index, y_pred, label=f'{model_name} Predicted',
                        linestyle='--', alpha=0.7)
            
            plt.title(title)
            plt.xlabel('Time')
            plt.ylabel('Energy Consumption')
            plt.legend()
            plt.grid(True)
            
            return plt.gcf()
            
        except Exception as e:
            raise ProcessingError(
                "Error plotting predictions",
                {'original_error': str(e)}
            )
    
    def plot_error_distribution(self, y_true: pd.Series,
                              predictions: Dict[str, np.ndarray]) -> plt.Figure:
        """Plot error distribution for multiple models"""
        try:
            n_models = len(predictions)
            fig, axes = plt.subplots(n_models, 1, figsize=(12, 4*n_models))
            
            if n_models == 1:
                axes = [axes]
            
            for (model_name, y_pred), ax in zip(predictions.items(), axes):
                errors = y_true - y_pred
                
                sns.histplot(errors, kde=True, ax=ax)
                ax.set_title(f'{model_name} Error Distribution')
                ax.set_xlabel('Error')
                ax.set_ylabel('Count')
                
                # Add statistical annotations
                mean_error = errors.mean()
                std_error = errors.std()
                ax.axvline(mean_error, color='r', linestyle='--',
                          label=f'Mean: {mean_error:.2f}')
                ax.axvline(mean_error + std_error, color='g', linestyle=':',
                          label=f'Std: {std_error:.2f}')
                ax.axvline(mean_error - std_error, color='g', linestyle=':')
                ax.legend()
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            raise ProcessingError(
                "Error plotting error distribution",
                {'original_error': str(e)}
            )
    
    def plot_feature_importance(self, feature_importance: pd.Series,
                              top_n: int = 10) -> plt.Figure:
        """Plot feature importance"""
        try:
            plt.figure(figsize=(10, 6))
            
            # Get top N features
            top_features = feature_importance.nlargest(top_n)
            
            # Create bar plot
            sns.barplot(x=top_features.values, y=top_features.index)
            
            plt.title(f'Top {top_n} Most Important Features')
            plt.xlabel('Importance Score')
            plt.tight_layout()
            
            return plt.gcf()
            
        except Exception as e:
            raise ProcessingError(
                "Error plotting feature importance",
                {'original_error': str(e)}
            )
    
    def analyze_residuals(self, y_true: pd.Series,
                         predictions: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Perform detailed residual analysis"""
        try:
            results = {}
            
            for model_name, y_pred in predictions.items():
                residuals = y_true - y_pred
                
                analysis = {
                    'mean_residual': residuals.mean(),
                    'std_residual': residuals.std(),
                    'skewness': pd.Series(residuals).skew(),
                    'kurtosis': pd.Series(residuals).kurtosis(),
                    'normality_test': self._test_normality(residuals)
                }
                
                # Test for heteroscedasticity
                analysis['heteroscedasticity'] = self._test_heteroscedasticity(
                    y_pred, residuals
                )
                
                results[model_name] = analysis
            
            return results
            
        except Exception as e:
            raise ProcessingError(
                "Error analyzing residuals",
                {'original_error': str(e)}
            )
    
    def _test_normality(self, residuals: np.ndarray) -> Dict[str, float]:
        """Test residuals for normality"""
        from scipy import stats
        
        statistic, p_value = stats.normaltest(residuals)
        return {
            'statistic': statistic,
            'p_value': p_value
        }
    
    def _test_heteroscedasticity(self, y_pred: np.ndarray,
                                residuals: np.ndarray) -> Dict[str, float]:
        """Test for heteroscedasticity using Breusch-Pagan test"""
        from scipy import stats
        
        # Fit residuals against predicted values
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            y_pred, residuals**2
        )
        
        return {
            'slope': slope,
            'p_value': p_value,
            'r_squared': r_value**2
        }
    
    def generate_evaluation_report(self, output_path: str) -> None:
        """Generate comprehensive evaluation report"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = f"{output_path}/evaluation_report_{timestamp}.html"
            
            # Create HTML report
            html_content = f"""
            <html>
            <head>
                <title>Model Evaluation Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    .section {{ margin: 20px 0; }}
                </style>
            </head>
            <body>
                <h1>Model Evaluation Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <div class="section">
                    <h2>Model Comparison</h2>
                    {self.compare_models(self.evaluation_results).to_html()}
                </div>
            </body>
            </html>
            """
            
            with open(report_path, 'w') as f:
                f.write(html_content)
                
        except Exception as e:
            raise ProcessingError(
                "Error generating evaluation report",
                {'output_path': output_path, 'original_error': str(e)}
            )
