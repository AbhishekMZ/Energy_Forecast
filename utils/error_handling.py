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

class BaseError(Exception):
    """Base error class with enhanced error information."""
    
    def __init__(self, message: str, error_code: str = None, 
                 details: Dict = None, recovery_hint: str = None):
        self.message = message
        self.error_code = error_code or 'UNKNOWN_ERROR'
        self.details = details or {}
        self.recovery_hint = recovery_hint
        self.timestamp = datetime.utcnow()
        super().__init__(self.message)
    
    def to_dict(self) -> Dict:
        """Convert error to dictionary format."""
        return {
            'error_code': self.error_code,
            'message': self.message,
            'details': self.details,
            'recovery_hint': self.recovery_hint,
            'timestamp': self.timestamp.isoformat(),
            'traceback': traceback.format_exc()
        }

class DataError(BaseError):
    """Data processing and validation errors."""
    pass

class ModelError(BaseError):
    """Model-related errors."""
    pass

class ConfigError(BaseError):
    """Configuration and parameter errors."""
    pass

class ResourceError(BaseError):
    """Resource allocation and management errors."""
    pass

class ErrorHandler:
    """Central error handling and recovery system."""
    
    def __init__(self):
        self.error_registry: Dict[str, Dict] = {}
        self.recovery_strategies: Dict[Type[BaseError], Callable] = {}
        self.fallback_strategies: Dict[Type[BaseError], Callable] = {}
    
    def register_recovery_strategy(self, error_type: Type[BaseError], 
                                 strategy: Callable, fallback: Callable = None):
        """Register recovery and fallback strategies for error types."""
        self.recovery_strategies[error_type] = strategy
        if fallback:
            self.fallback_strategies[error_type] = fallback
    
    def handle_error(self, error: BaseError, context: Dict = None) -> Any:
        """Handle error with appropriate recovery strategy."""
        error_info = error.to_dict()
        error_info['context'] = context or {}
        
        # Log error
        logger.error(f"Error occurred: {error_info['error_code']}", 
                    extra={'error_info': error_info})
        
        # Try recovery strategy
        if type(error) in self.recovery_strategies:
            try:
                logger.info(f"Attempting recovery for {error_info['error_code']}")
                return self.recovery_strategies[type(error)](error, context)
            except Exception as recovery_error:
                logger.error("Recovery failed", exc_info=recovery_error)
                
                # Try fallback
                if type(error) in self.fallback_strategies:
                    try:
                        logger.info("Attempting fallback strategy")
                        return self.fallback_strategies[type(error)](error, context)
                    except Exception as fallback_error:
                        logger.error("Fallback failed", exc_info=fallback_error)
        
        # If no recovery/fallback or all failed
        raise error

class ErrorMonitor:
    """Monitor and analyze error patterns."""
    
    def __init__(self):
        self.error_history: List[Dict] = []
        self.error_counts: Dict[str, int] = {}
    
    def record_error(self, error: BaseError):
        """Record error for analysis."""
        error_info = error.to_dict()
        self.error_history.append(error_info)
        self.error_counts[error.error_code] = self.error_counts.get(error.error_code, 0) + 1
    
    def get_error_statistics(self) -> Dict:
        """Get error statistics and patterns."""
        return {
            'total_errors': len(self.error_history),
            'error_counts': self.error_counts,
            'recent_errors': self.error_history[-10:],  # Last 10 errors
            'most_common': sorted(
                self.error_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]  # Top 5 most common errors
        }

def with_retry(max_retries: int = 3, delay: float = 1.0, 
               backoff: float = 2.0, exceptions: tuple = (Exception,)):
    """Decorator for automatic retry with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retry_count = 0
            current_delay = delay
            
            while retry_count < max_retries:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        raise
                    
                    logger.warning(
                        f"Attempt {retry_count} failed: {str(e)}. "
                        f"Retrying in {current_delay} seconds..."
                    )
                    
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            return None  # Should never reach here
        return wrapper
    return decorator

@contextmanager
def error_context(context_name: str, error_handler: ErrorHandler = None):
    """Context manager for error handling with context."""
    try:
        yield
    except BaseError as e:
        if error_handler:
            error_handler.handle_error(e, {'context': context_name})
        raise
    except Exception as e:
        logger.error(f"Unexpected error in {context_name}", exc_info=e)
        raise

class EnergyForecastException(BaseError):
    """Base exception class for Energy Forecast project"""
    def __init__(self, message: str, error_code: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)

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
