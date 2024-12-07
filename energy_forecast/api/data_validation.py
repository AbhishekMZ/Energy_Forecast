"""Data validation and quality checks for API requests"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

from energy_forecast.core.utils.data_quality import check_data_quality
from energy_forecast.core.validation.basic_validator import BasicDataValidator
from energy_forecast.core.validation.statistical_validator import StatisticalValidator
from energy_forecast.config.constants import VALIDATION_THRESHOLDS

logger = logging.getLogger(__name__)

class DataValidator:
    """Data validation for API requests"""
    
    def __init__(self):
        self.basic_validator = BasicDataValidator()
        self.statistical_validator = StatisticalValidator()
        
    def validate_forecast_request(
        self,
        data: Dict,
        city: str
    ) -> Tuple[bool, List[str]]:
        """Validate forecast request data"""
        try:
            errors = []
            
            # Convert to DataFrame for validation
            df = pd.DataFrame(data)
            
            # Basic validation
            basic_valid, basic_errors = self.basic_validator.run_all_validations(df)
            if not basic_valid:
                errors.extend(basic_errors)
            
            # Statistical validation
            stat_valid, stat_errors = self.statistical_validator.run_statistical_validations(df)
            if not stat_valid:
                errors.extend(stat_errors)
            
            # City-specific validation
            city_valid, city_errors = self._validate_city_specific(df, city)
            if not city_valid:
                errors.extend(city_errors)
            
            return len(errors) == 0, errors
            
        except Exception as e:
            logger.error(f"Error in forecast request validation: {str(e)}")
            return False, [str(e)]
    
    def _validate_city_specific(
        self,
        data: pd.DataFrame,
        city: str
    ) -> Tuple[bool, List[str]]:
        """Validate city-specific data patterns"""
        errors = []
        
        try:
            # Check data completeness
            missing_ratio = data.isnull().sum().max() / len(data)
            if missing_ratio > VALIDATION_THRESHOLDS['max_missing_ratio']:
                errors.append(f"Too many missing values: {missing_ratio:.2%}")
            
            # Check value ranges
            for col, limits in VALIDATION_THRESHOLDS.items():
                if col in data.columns:
                    out_of_range = data[
                        (data[col] < limits['min']) |
                        (data[col] > limits['max'])
                    ]
                    if len(out_of_range) > 0:
                        errors.append(
                            f"Values out of range for {col}: {len(out_of_range)} records"
                        )
            
            # Check temporal consistency
            if not data.index.is_monotonic_increasing:
                errors.append("Timestamps are not monotonically increasing")
            
            # Check for duplicates
            duplicates = data.index.duplicated()
            if duplicates.any():
                errors.append(f"Found {duplicates.sum()} duplicate timestamps")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            logger.error(f"Error in city-specific validation: {str(e)}")
            return False, [str(e)]
    
    def check_data_quality(
        self,
        data: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate data quality metrics"""
        metrics = {}
        
        try:
            # Completeness
            metrics['completeness'] = 1 - data.isnull().sum().mean() / len(data)
            
            # Consistency
            metrics['timestamp_consistency'] = float(data.index.is_monotonic_increasing)
            metrics['value_consistency'] = float(
                (data.select_dtypes(include=[np.number]) >= 0).all().all()
            )
            
            # Validity
            in_range = pd.Series(True, index=data.index)
            for col, limits in VALIDATION_THRESHOLDS.items():
                if col in data.columns:
                    in_range &= (
                        (data[col] >= limits['min']) &
                        (data[col] <= limits['max'])
                    )
            metrics['validity'] = float(in_range.mean())
            
            # Timeliness
            if len(data) > 0:
                latest_timestamp = data.index.max()
                time_lag = (datetime.now() - latest_timestamp).total_seconds() / 3600
                metrics['timeliness'] = max(0, 1 - time_lag / 24)  # Normalize to 24 hours
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating data quality metrics: {str(e)}")
            return {"error": str(e)}
