import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from scipy import stats
import logging
from datetime import datetime
from energy_forecast.core.validation.basic_validator import BasicDataValidator
from energy_forecast.core.utils.error_handling import ValidationError
from energy_forecast.config.constants import STATISTICAL_THRESHOLDS
from energy_forecast.core.utils.data_quality import check_data_quality

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StatisticalValidator:
    """Advanced statistical validation for energy forecast data"""
    
    def __init__(self):
        # Define expected correlations between features
        self.expected_correlations = {
            ('temperature', 'total_demand'): (0.3, 0.95),  # Strong positive correlation
            ('solar_radiation', 'total_demand'): (0.3, 0.95),  # Strong positive correlation
            ('cloud_cover', 'solar_radiation'): (-0.95, -0.3),  # Strong negative correlation
            ('temperature', 'humidity'): (-0.95, -0.3),  # Strong negative correlation
        }
        
        # Define seasonal patterns
        self.seasonal_patterns = {
            'Summer': [4, 5, 6],      # April to June
            'Monsoon': [7, 8, 9],     # July to September
            'Winter': [12, 1, 2]      # December to February
        }
        
        # Define distribution test parameters
        self.distribution_params = {
            'temperature': {'shapiro_threshold': 0.001},  # Less strict
            'humidity': {'shapiro_threshold': 0.001},
            'wind_speed': {'shapiro_threshold': 0.001}
        }
        
        # Define stationarity test parameters
        self.stationarity_params = {
            'mean_change_threshold': 1.5,  # Allow 150% variation
            'std_change_threshold': 1.5
        }
    
    def detect_outliers(self, df: pd.DataFrame, columns: List[str], 
                       threshold: float = 3) -> Dict[str, pd.Series]:
        """
        Detect outliers using Z-score method
        Returns dictionary with boolean masks for outliers in each column
        """
        outliers = {}
        
        for column in columns:
            if column in df.columns:
                # Calculate Z-scores
                z_scores = np.abs(stats.zscore(df[column]))
                outliers[column] = z_scores > threshold
        
        return outliers
    
    def validate_statistical_patterns(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate statistical patterns in the data"""
        errors = []
        
        # Check seasonal patterns
        for season, months in self.seasonal_patterns.items():
            season_data = df[df['timestamp'].dt.month.isin(months)]
            if len(season_data) == 0:
                continue  # Skip if no data for this season
            
            if season == 'Summer':
                # Check if demand increases with temperature in summer
                if 'temperature' in df.columns and 'total_demand' in df.columns:
                    corr = season_data['temperature'].corr(season_data['total_demand'])
                    if corr < 0.0:  # Just check for positive correlation
                        errors.append(f"Negative temperature-demand correlation in summer: {corr:.2f}")
            
            elif season == 'Monsoon':
                # Check if solar radiation is lower during monsoon
                if 'solar_radiation' in df.columns:
                    monsoon_solar = season_data['solar_radiation'].mean()
                    non_monsoon = df[~df['timestamp'].dt.month.isin(months)]
                    if len(non_monsoon) > 0:  # Check if we have non-monsoon data
                        non_monsoon_solar = non_monsoon['solar_radiation'].mean()
                        if monsoon_solar > non_monsoon_solar * 1.5:  # 50% higher is suspicious
                            errors.append("Unexpectedly high solar radiation during monsoon")
        
        # Check correlations only if we have enough data points
        min_data_points = 24  # At least a day's worth of data
        if len(df) >= min_data_points:
            for (feat1, feat2), (min_corr, max_corr) in self.expected_correlations.items():
                if feat1 in df.columns and feat2 in df.columns:
                    corr = df[feat1].corr(df[feat2])
                    # Add more tolerance to the correlation bounds
                    tolerance = 0.3  # Increased tolerance
                    if not (min_corr - tolerance) <= corr <= (max_corr + tolerance):
                        errors.append(f"Unexpected correlation between {feat1} and {feat2}: {corr:.2f}")
        
        return len(errors) == 0, errors
    
    def validate_distribution(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, List[str]]]:
        """Validate distribution patterns of key metrics"""
        errors = {}
        min_data_points = 24  # Minimum required data points
        
        if len(df) < min_data_points:
            return True, {}  # Not enough data for meaningful distribution analysis
        
        # Check demand distribution by hour
        if 'total_demand' in df.columns:
            hourly_demand = df.groupby(df['timestamp'].dt.hour)['total_demand'].mean()
            
            # Test for daily pattern (at least one peak)
            peaks = self._find_peaks(hourly_demand.values)
            if len(peaks) < 1:
                errors['demand_distribution'] = [
                    f"Expected at least one peak in hourly demand, found {len(peaks)} peaks"
                ]
        
        # Check for normal distribution of certain metrics
        for column in ['temperature', 'humidity', 'wind_speed']:
            if column in df.columns:
                # Get clean data sample
                clean_data = df[column].dropna()
                if len(clean_data) >= min_data_points:
                    # Use sample for large datasets
                    sample_size = min(1000, len(clean_data))
                    sample = clean_data.sample(sample_size)
                    
                    # Get threshold for this column
                    threshold = self.distribution_params[column]['shapiro_threshold']
                    
                    # Perform Shapiro-Wilk test for normality
                    try:
                        _, p_value = stats.shapiro(sample)
                        if p_value < threshold:
                            if column not in errors:
                                errors[column] = []
                            errors[column].append(f"Distribution is highly non-normal (p-value: {p_value:.4f})")
                    except Exception as e:
                        # Log but don't fail on statistical test errors
                        logging.warning(f"Error testing distribution for {column}: {str(e)}")
        
        return len(errors) == 0, errors
    
    def validate_stationarity(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, List[str]]]:
        """Check for stationarity in time series data"""
        errors = {}
        min_data_points = 24  # Minimum required data points
        
        if len(df) < min_data_points:
            return True, {}  # Not enough data for meaningful stationarity analysis
        
        # Group by city
        for city in df['city'].unique():
            city_data = df[df['city'] == city]
            
            # Check key metrics for stationarity
            for column in ['total_demand', 'temperature']:
                if column in df.columns:
                    # Simple stationarity check using rolling statistics
                    series = city_data[column]
                    rolling_mean = series.rolling(window=12).mean()
                    rolling_std = series.rolling(window=12).std()
                    
                    # Check if the rolling statistics are changing dramatically
                    mean_change = (rolling_mean.max() - rolling_mean.min()) / rolling_mean.mean()
                    std_change = (rolling_std.max() - rolling_std.min()) / rolling_std.mean()
                    
                    # Get thresholds
                    mean_threshold = self.stationarity_params['mean_change_threshold']
                    std_threshold = self.stationarity_params['std_change_threshold']
                    
                    if mean_change > mean_threshold or std_change > std_threshold:
                        if city not in errors:
                            errors[city] = []
                        errors[city].append(
                            f"{column} shows non-stationary behavior (mean_change: {mean_change:.2f}, std_change: {std_change:.2f})"
                        )
        
        return len(errors) == 0, errors
    
    def _find_peaks(self, data: np.array) -> List[int]:
        """Find peaks in the data"""
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(data, distance=4)  # Minimum 4 hours between peaks
        return peaks
    
    def run_statistical_validations(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, any]]:
        """Run all statistical validations"""
        validation_results = {}
        
        # Outlier detection
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        outliers = self.detect_outliers(df, numerical_columns)
        validation_results['outliers'] = {
            'valid': all(mask.sum() == 0 for mask in outliers.values()),
            'details': {col: mask.sum() for col, mask in outliers.items()}
        }
        
        # Statistical patterns
        patterns_valid, pattern_errors = self.validate_statistical_patterns(df)
        validation_results['patterns'] = {
            'valid': patterns_valid,
            'errors': pattern_errors
        }
        
        # Distribution validation
        dist_valid, dist_errors = self.validate_distribution(df)
        validation_results['distribution'] = {
            'valid': dist_valid,
            'errors': dist_errors
        }
        
        # Stationarity checks
        stat_valid, stat_errors = self.validate_stationarity(df)
        validation_results['stationarity'] = {
            'valid': stat_valid,
            'errors': stat_errors
        }
        
        # Overall validation result
        all_valid = all([
            validation_results['outliers']['valid'],
            patterns_valid,
            dist_valid,
            stat_valid
        ])
        
        return all_valid, validation_results
    
    def generate_statistical_report(self, validation_results: Dict) -> str:
        """Generate a formatted statistical validation report"""
        report = []
        report.append("=== Statistical Validation Report ===\n")
        
        # Overall status
        overall_valid = all(result['valid'] for result in validation_results.values())
        report.append(f"Overall Status: {'✓ PASSED' if overall_valid else '✗ FAILED'}\n")
        
        # Outlier summary
        report.append("Outlier Detection:")
        outliers = validation_results['outliers']['details']
        for column, count in outliers.items():
            report.append(f"  - {column}: {count} outliers detected")
        
        # Statistical patterns
        report.append("\nStatistical Patterns:")
        if not validation_results['patterns']['valid']:
            for error in validation_results['patterns']['errors']:
                report.append(f"  - {error}")
        else:
            report.append("  ✓ All patterns valid")
        
        # Distribution tests
        report.append("\nDistribution Tests:")
        if not validation_results['distribution']['valid']:
            for metric, errors in validation_results['distribution']['errors'].items():
                for error in errors:
                    report.append(f"  - {metric}: {error}")
        else:
            report.append("  ✓ All distributions valid")
        
        # Stationarity tests
        report.append("\nStationarity Tests:")
        if not validation_results['stationarity']['valid']:
            for city, errors in validation_results['stationarity']['errors'].items():
                for error in errors:
                    report.append(f"  - {city}: {error}")
        else:
            report.append("  ✓ All time series stationary")
        
        return "\n".join(report)
