from typing import List, Dict, Union, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import logging

class FeatureEngineer:
    """
    Advanced feature engineering pipeline for time series data
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scalers: Dict[str, Union[StandardScaler, MinMaxScaler]] = {}
        self.feature_importance: Dict[str, float] = {}
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive feature set from raw data
        """
        try:
            df_features = df.copy()
            
            # Time-based features
            df_features = self._create_time_features(df_features)
            
            # Weather-based features
            df_features = self._create_weather_features(df_features)
            
            # Statistical features
            df_features = self._create_statistical_features(df_features)
            
            # Interaction features
            df_features = self._create_interaction_features(df_features)
            
            # Lag features
            df_features = self._create_lag_features(df_features)
            
            return df_features
            
        except Exception as e:
            self.logger.error(f"Error in feature creation: {str(e)}")
            raise
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features
        """
        if 'timestamp' not in df.columns:
            return df
            
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Basic time features
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['year'] = df['timestamp'].dt.year
        df['dayofweek'] = df['timestamp'].dt.dayofweek
        df['quarter'] = df['timestamp'].dt.quarter
        
        # Cyclical features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        
        # Holiday features
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        # Add more holiday features as needed
        
        return df
    
    def _create_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create weather-based features
        """
        weather_columns = ['temperature', 'humidity', 'pressure', 'wind_speed']
        existing_columns = [col for col in weather_columns if col in df.columns]
        
        if not existing_columns:
            return df
            
        # Weather changes
        for col in existing_columns:
            df[f'{col}_change'] = df[col].diff()
            df[f'{col}_rolling_mean'] = df[col].rolling(window=24, min_periods=1).mean()
            df[f'{col}_rolling_std'] = df[col].rolling(window=24, min_periods=1).std()
            
        # Weather interactions
        if 'temperature' in df.columns and 'humidity' in df.columns:
            df['heat_index'] = self._calculate_heat_index(
                df['temperature'], df['humidity']
            )
            
        return df
    
    def _create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create statistical features
        """
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_columns:
            # Rolling statistics
            df[f'{col}_rolling_mean_7d'] = df[col].rolling(window=168, min_periods=1).mean()
            df[f'{col}_rolling_std_7d'] = df[col].rolling(window=168, min_periods=1).std()
            df[f'{col}_rolling_min_7d'] = df[col].rolling(window=168, min_periods=1).min()
            df[f'{col}_rolling_max_7d'] = df[col].rolling(window=168, min_periods=1).max()
            
            # Exponential moving averages
            df[f'{col}_ema_12h'] = df[col].ewm(span=12, adjust=False).mean()
            df[f'{col}_ema_24h'] = df[col].ewm(span=24, adjust=False).mean()
            
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between important variables
        """
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        main_features = ['temperature', 'humidity', 'total_load']
        
        for col1 in main_features:
            if col1 not in df.columns:
                continue
            for col2 in main_features:
                if col2 not in df.columns or col1 >= col2:
                    continue
                    
                # Multiplication interaction
                df[f'{col1}_{col2}_interaction'] = df[col1] * df[col2]
                
                # Ratio interaction
                if not (df[col2] == 0).any():
                    df[f'{col1}_{col2}_ratio'] = df[col1] / df[col2]
                    
        return df
    
    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create lag features for time series
        """
        target_cols = ['total_load', 'temperature']
        lag_periods = [1, 2, 3, 6, 12, 24, 48, 168]  # hours
        
        for col in target_cols:
            if col not in df.columns:
                continue
                
            for lag in lag_periods:
                df[f'{col}_lag_{lag}h'] = df[col].shift(lag)
                
        return df
    
    def _calculate_heat_index(self, temperature: pd.Series, humidity: pd.Series) -> pd.Series:
        """
        Calculate heat index using temperature and humidity
        """
        T = temperature * 9/5 + 32  # Convert to Fahrenheit
        RH = humidity
        
        heat_index = 0.5 * (T + 61.0 + ((T-68.0)*1.2) + (RH*0.094))
        
        mask = T >= 80
        heat_index[mask] = -42.379 + 2.04901523*T + 10.14333127*RH - \
                          0.22475541*T*RH - 6.83783e-3*T**2 - \
                          5.481717e-2*RH**2 + 1.22874e-3*T**2*RH + \
                          8.5282e-4*T*RH**2 - 1.99e-6*T**2*RH**2
                          
        return (heat_index - 32) * 5/9  # Convert back to Celsius
    
    def scale_features(self, df: pd.DataFrame, feature_cols: List[str], 
                      scaler_type: str = 'standard') -> pd.DataFrame:
        """
        Scale features using specified scaler
        """
        df_scaled = df.copy()
        
        for col in feature_cols:
            if col not in df.columns:
                continue
                
            if col not in self.scalers:
                if scaler_type == 'standard':
                    self.scalers[col] = StandardScaler()
                else:
                    self.scalers[col] = MinMaxScaler()
                    
            df_scaled[col] = self.scalers[col].fit_transform(
                df[col].values.reshape(-1, 1)
            )
            
        return df_scaled
    
    def reduce_dimensions(self, df: pd.DataFrame, n_components: int = 10) -> pd.DataFrame:
        """
        Perform dimensionality reduction using PCA
        """
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        pca = PCA(n_components=min(n_components, len(numerical_cols)))
        
        pca_result = pca.fit_transform(df[numerical_cols])
        
        # Create PCA features
        pca_df = pd.DataFrame(
            pca_result,
            columns=[f'pca_{i+1}' for i in range(pca_result.shape[1])],
            index=df.index
        )
        
        # Store feature importance
        for i, importance in enumerate(pca.explained_variance_ratio_):
            self.feature_importance[f'pca_{i+1}'] = importance
            
        return pca_df
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores
        """
        return self.feature_importance
