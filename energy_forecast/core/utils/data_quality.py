from typing import Dict, List, Union, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from scipy import stats

class DataQualityChecker:
    """
    Advanced data quality checking system with statistical analysis
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.quality_metrics: Dict[str, Dict] = {}
        self.historical_stats: Dict[str, List[float]] = {}
        
    def check_data_quality(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Comprehensive data quality check
        Returns quality metrics for each column
        """
        quality_report = {
            'overall_quality': {},
            'column_metrics': {},
            'correlations': {},
            'anomalies': {},
            'recommendations': []
        }
        
        # Basic quality checks
        quality_report['overall_quality'] = self._check_basic_quality(df)
        
        # Column-specific checks
        for column in df.columns:
            quality_report['column_metrics'][column] = self._analyze_column(df[column])
            
        # Correlation analysis
        quality_report['correlations'] = self._analyze_correlations(df)
        
        # Anomaly detection
        quality_report['anomalies'] = self._detect_anomalies(df)
        
        # Generate recommendations
        quality_report['recommendations'] = self._generate_recommendations(quality_report)
        
        return quality_report
    
    def _check_basic_quality(self, df: pd.DataFrame) -> Dict:
        """
        Perform basic quality checks on the entire dataset
        """
        total_rows = len(df)
        return {
            'total_rows': total_rows,
            'total_columns': len(df.columns),
            'missing_cells': df.isna().sum().sum(),
            'missing_cells_ratio': df.isna().sum().sum() / (total_rows * len(df.columns)),
            'duplicate_rows': df.duplicated().sum(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
            'data_types': df.dtypes.value_counts().to_dict()
        }
    
    def _analyze_column(self, series: pd.Series) -> Dict:
        """
        Detailed analysis of individual columns
        """
        analysis = {
            'data_type': str(series.dtype),
            'missing_values': series.isna().sum(),
            'unique_values': series.nunique(),
            'memory_usage': series.memory_usage(deep=True) / 1024**2  # MB
        }
        
        # Numerical column analysis
        if pd.api.types.is_numeric_dtype(series):
            analysis.update({
                'mean': series.mean(),
                'median': series.median(),
                'std': series.std(),
                'min': series.min(),
                'max': series.max(),
                'skewness': series.skew(),
                'kurtosis': series.kurtosis(),
                'outliers': self._detect_column_outliers(series)
            })
            
        # Temporal column analysis
        elif pd.api.types.is_datetime64_any_dtype(series):
            analysis.update({
                'min_date': series.min(),
                'max_date': series.max(),
                'date_range': (series.max() - series.min()).days,
                'missing_dates': self._check_missing_dates(series)
            })
            
        # Categorical column analysis
        elif pd.api.types.is_string_dtype(series):
            value_counts = series.value_counts()
            analysis.update({
                'most_common': value_counts.index[0] if not value_counts.empty else None,
                'least_common': value_counts.index[-1] if not value_counts.empty else None,
                'empty_strings': (series == '').sum(),
                'whitespace_only': series.str.isspace().sum() if hasattr(series.str, 'isspace') else 0
            })
            
        return analysis
    
    def _detect_column_outliers(self, series: pd.Series) -> Dict:
        """
        Detect outliers using multiple methods
        """
        # Z-score method
        z_scores = np.abs(stats.zscore(series.dropna()))
        z_score_outliers = len(z_scores[z_scores > 3])
        
        # IQR method
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        iqr_outliers = len(series[(series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))])
        
        return {
            'z_score_outliers': z_score_outliers,
            'iqr_outliers': iqr_outliers,
            'total_outliers': max(z_score_outliers, iqr_outliers)
        }
    
    def _check_missing_dates(self, series: pd.Series) -> Dict:
        """
        Check for missing dates in time series data
        """
        if not pd.api.types.is_datetime64_any_dtype(series):
            return {}
            
        date_range = pd.date_range(start=series.min(), end=series.max(), freq='D')
        missing_dates = date_range.difference(series.unique())
        
        return {
            'missing_dates_count': len(missing_dates),
            'missing_dates': missing_dates.tolist() if len(missing_dates) < 10 else missing_dates[:10].tolist()
        }
    
    def _analyze_correlations(self, df: pd.DataFrame) -> Dict:
        """
        Analyze correlations between numerical columns
        """
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) < 2:
            return {}
            
        corr_matrix = df[numerical_cols].corr()
        
        # Find highly correlated pairs
        high_correlations = []
        for i in range(len(numerical_cols)):
            for j in range(i+1, len(numerical_cols)):
                correlation = corr_matrix.iloc[i, j]
                if abs(correlation) > 0.7:  # Threshold for high correlation
                    high_correlations.append({
                        'columns': (numerical_cols[i], numerical_cols[j]),
                        'correlation': correlation
                    })
                    
        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'high_correlations': high_correlations
        }
    
    def _detect_anomalies(self, df: pd.DataFrame) -> Dict:
        """
        Detect various types of anomalies in the dataset
        """
        anomalies = {
            'sudden_changes': {},
            'pattern_breaks': {},
            'unusual_distributions': {}
        }
        
        for column in df.select_dtypes(include=[np.number]).columns:
            # Detect sudden changes using rolling statistics
            rolling_mean = df[column].rolling(window=7).mean()
            rolling_std = df[column].rolling(window=7).std()
            
            # Flag values that are more than 3 standard deviations from the rolling mean
            anomalies['sudden_changes'][column] = len(
                df[abs(df[column] - rolling_mean) > 3 * rolling_std]
            )
            
            # Check for unusual distributions
            if len(df[column].dropna()) > 30:  # Need sufficient data points
                _, p_value = stats.normaltest(df[column].dropna())
                anomalies['unusual_distributions'][column] = {
                    'normal_distribution_p_value': p_value,
                    'is_normal': p_value > 0.05
                }
                
        return anomalies
    
    def _generate_recommendations(self, quality_report: Dict) -> List[str]:
        """
        Generate recommendations based on quality analysis
        """
        recommendations = []
        
        # Missing data recommendations
        if quality_report['overall_quality']['missing_cells_ratio'] > 0.1:
            recommendations.append(
                "High proportion of missing data. Consider imputation techniques."
            )
            
        # Outlier recommendations
        for column, metrics in quality_report['column_metrics'].items():
            if 'outliers' in metrics and metrics['outliers']['total_outliers'] > 0:
                recommendations.append(
                    f"Column '{column}' contains {metrics['outliers']['total_outliers']} "
                    "outliers. Consider investigating or applying outlier treatment."
                )
                
        # Correlation recommendations
        if 'high_correlations' in quality_report['correlations']:
            for corr in quality_report['correlations']['high_correlations']:
                recommendations.append(
                    f"High correlation ({corr['correlation']:.2f}) between "
                    f"{corr['columns'][0]} and {corr['columns'][1]}. "
                    "Consider feature selection or dimensionality reduction."
                )
                
        return recommendations
    
    def export_report(self, report: Dict, filename: str):
        """
        Export quality report to JSON file
        """
        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            return True
        except Exception as e:
            self.logger.error(f"Error exporting report: {str(e)}")
            return False
