"""Specialized energy forecasting model with seasonal and weather considerations"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from datetime import datetime, timedelta
import logging

from energy_forecast.core.models.pipeline import MLPipeline
from energy_forecast.core.optimization.renewable_optimizer import RenewableOptimizer
from energy_forecast.config.constants import CITIES, RENEWABLE_SOURCES
from energy_forecast.core.utils.error_handling import ProcessingError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnergyForecastModel(MLPipeline):
    """Specialized model for energy demand forecasting with renewable optimization"""
    
    def __init__(self, city: str, config: Dict = None):
        super().__init__(config)
        self.city = city
        self.city_data = CITIES[city]
        self.renewable_optimizer = RenewableOptimizer(city)
        self.scaler = StandardScaler()
        
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare specialized features for energy forecasting"""
        try:
            df = data.copy()
            
            # Time-based features
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month
            df['season'] = self._get_season(df.index)
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            df['is_holiday'] = self._is_holiday(df.index)
            
            # Seasonal decomposition
            df['yearly_trend'] = self._calculate_yearly_trend(df['total_demand'])
            df['seasonal_factor'] = self._calculate_seasonal_factor(df['total_demand'])
            
            # Weather impact features
            weather_cols = ['temperature', 'humidity', 'cloud_cover', 'solar_radiation',
                          'wind_speed', 'precipitation']
            
            for col in weather_cols:
                if col in df.columns:
                    # Current values
                    df[f'{col}_scaled'] = self.scaler.fit_transform(df[[col]])
                    
                    # Rolling statistics
                    df[f'{col}_rolling_mean_24h'] = df[col].rolling(24).mean()
                    df[f'{col}_rolling_std_24h'] = df[col].rolling(24).std()
                    
                    # Rate of change
                    df[f'{col}_change_rate'] = df[col].diff()
            
            # Demand features
            df['demand_lag_1h'] = df['total_demand'].shift(1)
            df['demand_lag_24h'] = df['total_demand'].shift(24)
            df['demand_lag_168h'] = df['total_demand'].shift(168)  # 1 week
            
            # Population-based scaling
            df['demand_per_capita'] = df['total_demand'] / self.city_data['population']
            
            # Remove missing values
            df = df.dropna()
            
            return df
            
        except Exception as e:
            raise ProcessingError(
                "Error preparing features",
                {'original_error': str(e)}
            )
    
    def train(
        self,
        data: pd.DataFrame,
        validation_split: float = 0.2
    ) -> Dict[str, float]:
        """Train the energy forecasting model"""
        try:
            # Prepare features
            features_df = self.prepare_features(data)
            
            # Split features and target
            X = features_df.drop(['total_demand'], axis=1)
            y = features_df['total_demand']
            
            # Train-validation split
            train_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:train_idx], X[train_idx:]
            y_train, y_val = y[:train_idx], y[train_idx:]
            
            # Train XGBoost model
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=7,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            
            # Early stopping using validation set
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='rmse',
                early_stopping_rounds=50,
                verbose=False
            )
            
            self.models['xgboost'] = model
            
            # Calculate and store feature importance
            importance = pd.Series(
                model.feature_importances_,
                index=X.columns
            ).sort_values(ascending=False)
            
            self.feature_importance['xgboost'] = importance
            
            # Evaluate model
            y_pred = model.predict(X_val)
            metrics = self.evaluate_model('xgboost', y_val, pd.Series(y_pred))
            
            return metrics
            
        except Exception as e:
            raise ProcessingError(
                "Error training model",
                {'original_error': str(e)}
            )
    
    def forecast_demand(
        self,
        current_data: pd.DataFrame,
        forecast_horizon: int = 168  # 1 week
    ) -> pd.DataFrame:
        """Generate demand forecast with confidence intervals"""
        try:
            forecast_dates = pd.date_range(
                start=current_data.index[-1],
                periods=forecast_horizon + 1,
                freq='H'
            )[1:]
            
            # Initialize forecast dataframe
            forecast = pd.DataFrame(index=forecast_dates)
            
            # Generate features for forecast period
            forecast_features = self._generate_forecast_features(
                current_data,
                forecast_dates
            )
            
            # Make predictions
            forecast['demand_mean'] = self.models['xgboost'].predict(forecast_features)
            
            # Add confidence intervals (using historical volatility)
            volatility = current_data['total_demand'].std()
            forecast['demand_lower'] = forecast['demand_mean'] - 1.96 * volatility
            forecast['demand_upper'] = forecast['demand_mean'] + 1.96 * volatility
            
            return forecast
            
        except Exception as e:
            raise ProcessingError(
                "Error generating forecast",
                {'original_error': str(e)}
            )
    
    def optimize_energy_mix(
        self,
        demand_forecast: pd.DataFrame,
        weather_forecast: pd.DataFrame
    ) -> pd.DataFrame:
        """Optimize energy mix for forecasted demand"""
        try:
            results = []
            
            for timestamp in demand_forecast.index:
                demand = demand_forecast.loc[timestamp, 'demand_mean']
                weather = weather_forecast.loc[timestamp]
                
                # Get current capacity for each source
                current_capacity = self.city_data['energy_sources']
                
                # Optimize energy mix
                energy_mix = self.renewable_optimizer.optimize_energy_mix(
                    demand,
                    weather,
                    current_capacity
                )
                
                # Get production schedule
                schedule = self.renewable_optimizer.get_production_schedule(
                    energy_mix,
                    timestamp
                )
                
                # Calculate total cost
                total_cost = sum(
                    amount * RENEWABLE_SOURCES[source]['cost_per_mwh']
                    for source, amount in energy_mix.items()
                )
                
                results.append({
                    'timestamp': timestamp,
                    'demand': demand,
                    'energy_mix': energy_mix,
                    'schedule': schedule,
                    'total_cost': total_cost
                })
            
            return pd.DataFrame(results)
            
        except Exception as e:
            raise ProcessingError(
                "Error optimizing energy mix",
                {'original_error': str(e)}
            )
    
    def _get_season(self, dates: pd.DatetimeIndex) -> pd.Series:
        """Determine season for given dates"""
        # Adjust for Northern Hemisphere
        return pd.Series(dates.month % 12 // 3).map({
            0: 'winter',
            1: 'spring',
            2: 'summer',
            3: 'fall'
        })
    
    def _is_holiday(self, dates: pd.DatetimeIndex) -> pd.Series:
        """Determine if dates are holidays"""
        # Implement holiday detection logic here
        # For now, just return weekends
        return pd.Series(dates.dayofweek).isin([5, 6]).astype(int)
    
    def _calculate_yearly_trend(self, series: pd.Series) -> pd.Series:
        """Calculate yearly trend component"""
        # Use rolling average to capture long-term trend
        return series.rolling(window=8760, min_periods=1, center=True).mean()
    
    def _calculate_seasonal_factor(self, series: pd.Series) -> pd.Series:
        """Calculate seasonal factors"""
        # Use rolling average to capture seasonal patterns
        seasonal = series.rolling(window=168, min_periods=1, center=True).mean()
        return seasonal / self._calculate_yearly_trend(series)
    
    def _generate_forecast_features(
        self,
        historical_data: pd.DataFrame,
        forecast_dates: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """Generate features for forecast period"""
        # Create empty dataframe with forecast dates
        forecast = pd.DataFrame(index=forecast_dates)
        
        # Add time-based features
        forecast['hour'] = forecast.index.hour
        forecast['day_of_week'] = forecast.index.dayofweek
        forecast['month'] = forecast.index.month
        forecast['season'] = self._get_season(forecast.index)
        forecast['is_weekend'] = forecast['day_of_week'].isin([5, 6]).astype(int)
        forecast['is_holiday'] = self._is_holiday(forecast.index)
        
        # Add lagged features from historical data
        last_demand = historical_data['total_demand'].iloc[-1]
        forecast['demand_lag_1h'] = last_demand
        
        # Add other necessary features
        # This will depend on your model's feature requirements
        
        return forecast
