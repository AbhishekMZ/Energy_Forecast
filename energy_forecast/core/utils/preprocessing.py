from typing import Dict, List, Union, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import logging
from datetime import datetime, timedelta
from .error_handling import (
    ProcessingError, error_handler, validate_dataframe,
    DataValidator
)

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for energy forecasting
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.imputers: Dict[str, Union[SimpleImputer, KNNImputer]] = {}
        self.scalers: Dict[str, Union[StandardScaler, RobustScaler, MinMaxScaler]] = {}
        self.column_stats: Dict[str, Dict] = {}
        self.validator = DataValidator()
        
    @error_handler
    def preprocess_data(self, df: pd.DataFrame, 
                       config: Optional[Dict] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Main preprocessing pipeline with enhanced error handling
        """
        try:
            df_processed = df.copy()
            validation_report = {}
            
            # Validate input data
            required_cols = ['timestamp', 'temperature', 'humidity', 'consumption']
            numeric_cols = ['temperature', 'humidity', 'consumption']
            datetime_cols = ['timestamp']
            
            validate_dataframe(
                df_processed,
                required_columns=required_cols,
                numeric_columns=numeric_cols,
                datetime_columns=datetime_cols
            )
            
            # Check data quality
            validation_report['missing_values'] = self.validator.check_missing_values(
                df_processed
            )
            validation_report['outliers'] = self.validator.check_outliers(
                df_processed, numeric_cols
            )
            
            # Store original data stats
            self._store_column_stats(df_processed)
            
            # Handle missing values
            df_processed = self._handle_missing_values(df_processed, config)
            
            # Handle outliers
            df_processed = self._handle_outliers(df_processed, config)
            
            # Format timestamps
            df_processed = self._format_timestamps(df_processed)
            
            # Scale numerical features
            df_processed = self._scale_features(df_processed, config)
            
            # Final validation
            validation_report['final_check'] = self._validate_processed_data(
                df_processed
            )
            
            return df_processed, validation_report
            
        except Exception as e:
            raise ProcessingError(
                "Error in preprocessing pipeline",
                {'original_error': str(e)}
            )
    
    @error_handler
    def _handle_missing_values(self, df: pd.DataFrame, 
                             config: Optional[Dict] = None) -> pd.DataFrame:
        """
        Handle missing values with multiple strategies
        """
        try:
            config = config or {}
            impute_strategy = config.get('impute_strategy', 'knn')
            
            for column in df.columns:
                if df[column].isna().any():
                    if column not in self.imputers:
                        if impute_strategy == 'knn':
                            self.imputers[column] = KNNImputer(n_neighbors=5)
                        else:
                            self.imputers[column] = SimpleImputer(
                                strategy=impute_strategy
                            )
                            
                    values = df[column].values.reshape(-1, 1)
                    df[column] = self.imputers[column].fit_transform(values).ravel()
                    
            return df
            
        except Exception as e:
            raise ProcessingError(
                "Error handling missing values",
                {'strategy': impute_strategy, 'original_error': str(e)}
            )
    
    @error_handler
    def _handle_outliers(self, df: pd.DataFrame, 
                        config: Optional[Dict] = None) -> pd.DataFrame:
        """
        Handle outliers with configurable methods
        """
        try:
            config = config or {}
            outlier_method = config.get('outlier_method', 'iqr')
            threshold = config.get('outlier_threshold', 1.5)
            
            for column in df.select_dtypes(include=[np.number]).columns:
                if outlier_method == 'iqr':
                    df = self._handle_outliers_iqr(df, column, threshold)
                elif outlier_method == 'zscore':
                    df = self._handle_outliers_zscore(df, column, threshold)
                    
            return df
            
        except Exception as e:
            raise ProcessingError(
                "Error handling outliers",
                {
                    'method': outlier_method,
                    'threshold': threshold,
                    'original_error': str(e)
                }
            )
    
    @error_handler
    def _format_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Format and validate timestamps
        """
        try:
            timestamp_columns = df.select_dtypes(include=['datetime64']).columns
            
            for column in timestamp_columns:
                # Ensure timezone awareness
                if df[column].dt.tz is None:
                    df[column] = df[column].dt.tz_localize('UTC')
                    
                # Check for gaps
                time_diff = df[column].diff()
                if time_diff.max() > timedelta(hours=1):
                    self.logger.warning(
                        f"Found gaps in timestamp column {column}"
                    )
                    
            return df
            
        except Exception as e:
            raise ProcessingError(
                "Error formatting timestamps",
                {'original_error': str(e)}
            )
    
    @error_handler
    def _scale_features(self, df: pd.DataFrame, 
                       config: Optional[Dict] = None) -> pd.DataFrame:
        """
        Scale numerical features with error handling
        """
        try:
            config = config or {}
            scaler_type = config.get('scaler', 'standard')
            numerical_columns = df.select_dtypes(include=[np.number]).columns
            
            for column in numerical_columns:
                if column not in self.scalers:
                    if scaler_type == 'robust':
                        self.scalers[column] = RobustScaler()
                    elif scaler_type == 'minmax':
                        self.scalers[column] = MinMaxScaler()
                    else:
                        self.scalers[column] = StandardScaler()
                        
                values = df[column].values.reshape(-1, 1)
                df[column] = self.scalers[column].fit_transform(values).ravel()
                
            return df
            
        except Exception as e:
            raise ProcessingError(
                "Error scaling features",
                {'scaler_type': scaler_type, 'original_error': str(e)}
            )
    
    @error_handler
    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform scaled features
        """
        try:
            df_inverse = df.copy()
            
            for column, scaler in self.scalers.items():
                if column in df_inverse.columns:
                    values = df_inverse[column].values.reshape(-1, 1)
                    df_inverse[column] = scaler.inverse_transform(values).ravel()
                    
            return df_inverse
            
        except Exception as e:
            raise ProcessingError(
                "Error performing inverse transform",
                {'original_error': str(e)}
            )
    
    def _validate_processed_data(self, df: pd.DataFrame) -> Dict:
        """
        Validate processed data and generate report
        """
        validation_issues = {
            'missing_values': df.isna().sum().to_dict(),
            'infinity_values': np.isinf(
                df.select_dtypes(include=[np.number])
            ).sum().to_dict(),
            'negative_values': (
                df.select_dtypes(include=[np.number]) < 0
            ).sum().to_dict(),
            'column_types': df.dtypes.to_dict(),
            'shape': df.shape
        }
        
        if any(validation_issues['missing_values'].values()):
            self.logger.warning("Found remaining missing values after preprocessing")
        if any(validation_issues['infinity_values'].values()):
            self.logger.warning("Found infinity values after preprocessing")
            
        return validation_issues
    
    def get_preprocessing_summary(self) -> Dict:
        """
        Get summary of preprocessing operations
        """
        return {
            'original_stats': self.column_stats,
            'imputers': {k: str(v) for k, v in self.imputers.items()},
            'scalers': {k: str(v) for k, v in self.scalers.items()}
        }
