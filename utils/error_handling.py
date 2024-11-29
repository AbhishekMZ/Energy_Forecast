from typing import Optional, Dict, Any
import logging
import traceback
from functools import wraps
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class EnergyForecastException(Exception):
    """Base exception class for Energy Forecast project"""
    def __init__(self, message: str, error_code: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

class DataValidationError(EnergyForecastException):
    """Raised when data validation fails"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "DATA_VALIDATION_ERROR", details)

class ProcessingError(EnergyForecastException):
    """Raised when data processing fails"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "PROCESSING_ERROR", details)

class DatabaseError(EnergyForecastException):
    """Raised when database operations fail"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "DATABASE_ERROR", details)

class VisualizationError(EnergyForecastException):
    """Raised when visualization operations fail"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "VISUALIZATION_ERROR", details)

def log_error(logger: logging.Logger, error: Exception, context: Optional[Dict[str, Any]] = None):
    """Centralized error logging function"""
    error_details = {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'traceback': traceback.format_exc(),
        'context': context or {}
    }
    logger.error(f"Error occurred: {error_details['error_type']}", extra=error_details)
    return error_details

def validate_dataframe(df: pd.DataFrame, required_columns: Optional[list] = None, 
                      numeric_columns: Optional[list] = None,
                      datetime_columns: Optional[list] = None) -> None:
    """Validate DataFrame structure and content"""
    try:
        # Check if DataFrame is empty
        if df.empty:
            raise DataValidationError("DataFrame is empty")

        # Validate required columns
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                raise DataValidationError(
                    f"Missing required columns: {missing_cols}",
                    {'missing_columns': list(missing_cols)}
                )

        # Validate numeric columns
        if numeric_columns:
            for col in numeric_columns:
                if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                    raise DataValidationError(
                        f"Column {col} must be numeric",
                        {'column': col, 'current_type': str(df[col].dtype)}
                    )

        # Validate datetime columns
        if datetime_columns:
            for col in datetime_columns:
                if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
                    raise DataValidationError(
                        f"Column {col} must be datetime",
                        {'column': col, 'current_type': str(df[col].dtype)}
                    )

    except DataValidationError:
        raise
    except Exception as e:
        raise DataValidationError("Unexpected error during data validation", 
                                {'original_error': str(e)})

def error_handler(func):
    """Decorator for consistent error handling"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        try:
            return func(*args, **kwargs)
        except EnergyForecastException as e:
            log_error(logger, e)
            raise
        except Exception as e:
            error_details = log_error(logger, e)
            raise EnergyForecastException(
                f"Unexpected error in {func.__name__}",
                "UNEXPECTED_ERROR",
                error_details
            )
    return wrapper

class DataValidator:
    """Class for validating data quality and integrity"""
    
    @staticmethod
    def check_missing_values(df: pd.DataFrame, threshold: float = 0.1) -> Dict[str, Any]:
        """Check for missing values in DataFrame"""
        missing_stats = {
            'total_missing': df.isna().sum().to_dict(),
            'missing_percentage': (df.isna().sum() / len(df) * 100).to_dict()
        }
        
        # Identify columns with missing values above threshold
        problematic_columns = {
            col: pct for col, pct in missing_stats['missing_percentage'].items()
            if pct > threshold * 100
        }
        
        if problematic_columns:
            raise DataValidationError(
                f"Columns exceed missing value threshold of {threshold*100}%",
                {'problematic_columns': problematic_columns}
            )
            
        return missing_stats

    @staticmethod
    def check_outliers(df: pd.DataFrame, columns: list, 
                      method: str = 'iqr', threshold: float = 1.5) -> Dict[str, Any]:
        """Check for outliers in specified columns"""
        outlier_stats = {}
        
        for col in columns:
            if col not in df.columns:
                continue
                
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
                
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = df[
                    (df[col] < lower_bound) | 
                    (df[col] > upper_bound)
                ]
            else:  # z-score method
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = df[z_scores > threshold]
            
            if not outliers.empty:
                outlier_stats[col] = {
                    'count': len(outliers),
                    'percentage': len(outliers) / len(df) * 100,
                    'values': outliers[col].tolist()
                }
        
        if outlier_stats:
            raise DataValidationError(
                "Outliers detected in data",
                {'outlier_statistics': outlier_stats}
            )
            
        return outlier_stats

    @staticmethod
    def check_data_consistency(df: pd.DataFrame, rules: Dict[str, Any]) -> Dict[str, Any]:
        """Check data consistency based on provided rules"""
        consistency_issues = {}
        
        for rule_name, rule_config in rules.items():
            column = rule_config.get('column')
            condition = rule_config.get('condition')
            
            if not column or not condition:
                continue
                
            invalid_data = df[~df[column].apply(condition)]
            
            if not invalid_data.empty:
                consistency_issues[rule_name] = {
                    'count': len(invalid_data),
                    'percentage': len(invalid_data) / len(df) * 100,
                    'sample_values': invalid_data[column].head().tolist()
                }
        
        if consistency_issues:
            raise DataValidationError(
                "Data consistency checks failed",
                {'consistency_issues': consistency_issues}
            )
            
        return consistency_issues
