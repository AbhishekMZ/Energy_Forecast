import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime

# Update imports to use absolute imports from energy_forecast package
from energy_forecast.core.utils.data_quality import check_data_quality
from energy_forecast.core.utils.error_handling import ValidationError, ProcessingError
from energy_forecast.core.utils.feature_engineering import (
    create_time_features,
    create_weather_features,
    create_demand_features
)
from energy_forecast.config.constants import PIPELINE_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPipeline:
    """Data processing pipeline for energy forecast data"""
    
    def __init__(self, config: Dict = None):
        self.config = config or PIPELINE_CONFIG
        
    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process input data through the pipeline"""
        try:
            # Validate input data
            valid, errors = check_data_quality(data)
            if not valid:
                raise ValidationError("Data quality check failed", errors)
                
            # Create features
            data = self._create_features(data)
            
            # Handle missing values
            data = self._handle_missing_values(data)
            
            # Scale features
            data = self._scale_features(data)
            
            return data
            
        except Exception as e:
            raise ProcessingError(
                "Error in data pipeline",
                {'original_error': str(e)}
            )
            
    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create all required features"""
        data = create_time_features(data)
        data = create_weather_features(data)
        data = create_demand_features(data)
        return data
        
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values based on config"""
        for col in data.columns:
            if col in self.config['fill_methods']:
                method = self.config['fill_methods'][col]
                if method == 'mean':
                    data[col].fillna(data[col].mean(), inplace=True)
                elif method == 'median':
                    data[col].fillna(data[col].median(), inplace=True)
                elif method == 'interpolate':
                    data[col].interpolate(method='linear', inplace=True)
                    
        return data
        
    def _scale_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Scale features based on config"""
        for col in self.config['scaling']:
            if self.config['scaling'][col] == 'standard':
                data[col] = (data[col] - data[col].mean()) / data[col].std()
            elif self.config['scaling'][col] == 'minmax':
                data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
                
        return data
