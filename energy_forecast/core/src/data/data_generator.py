import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from ..utils.data_loader import HistoricalDataLoader
from ..config.constants import (
    INDIAN_SEASONS, INDIAN_HOLIDAYS, PEAK_HOURS,
    CITY_BASELINES, INFRASTRUCTURE_CONFIG, WEATHER_PATTERNS
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealisticIndianEnergyGenerator:
    def __init__(self, start_date: datetime, end_date: datetime):
        self.start_date = start_date
        self.end_date = end_date
        self.historical_loader = HistoricalDataLoader()
        self.cities = list(CITY_BASELINES.keys())
        
        # Load historical data and patterns
        self.load_historical_data()
        
    def load_historical_data(self):
        """Load historical weather and energy data for all cities"""
        self.historical_weather = {}
        self.energy_patterns = {}
        self.infrastructure = {}
        
        for city in self.cities:
            logger.info(f"Loading historical data for {city}")
            
            # Load 5 years of historical weather data
            historical_start = datetime(self.start_date.year - 5, 1, 1)
            self.historical_weather[city] = self.historical_loader.fetch_historical_weather(
                city, historical_start.year, self.start_date.year - 1
            )
            
            # Load city-specific patterns
            self.energy_patterns[city] = CITY_BASELINES[city]
            self.infrastructure[city] = INFRASTRUCTURE_CONFIG[city]
    
    def generate_weather_features(self, timestamp: datetime, city: str) -> Dict:
        """Generate weather features based on historical patterns"""
        # Find similar historical dates (same month, similar hour)
        historical_data = self.historical_weather[city]
        similar_conditions = historical_data[
            (historical_data['timestamp'].dt.month == timestamp.month) &
            (historical_data['timestamp'].dt.hour.between(timestamp.hour - 1, timestamp.hour + 1))
        ]
        
        if len(similar_conditions) > 0:
            # Sample from historical data with some random variation
            sample = similar_conditions.sample(1).iloc[0]
            return {
                'temperature': sample['temperature'] + np.random.normal(0, 0.5),
                'humidity': np.clip(sample['humidity'] + np.random.normal(0, 2), 0, 100),
                'precipitation': max(0, sample['precipitation'] + np.random.exponential(1) if sample['precipitation'] > 0 else 0),
                'solar_radiation': max(0, sample['solar_radiation'] + np.random.normal(0, 50)),
                'cloud_cover': np.clip(sample['cloud_cover'] + np.random.normal(0, 5), 0, 100),
                'wind_speed': max(0, sample['wind_speed'] + np.random.normal(0, 1)),
                'wind_direction': sample['wind_direction']
            }
        else:
            # Fallback to synthetic generation
            return self._synthetic_weather(timestamp, city)
    
    def _synthetic_weather(self, timestamp: datetime, city: str) -> Dict:
        """Fallback method for synthetic weather generation"""
        pattern = WEATHER_PATTERNS[city]
        month = timestamp.month
        hour = timestamp.hour
        
        is_monsoon = month in pattern['rainfall_months']
        temp_min, temp_max = pattern['temp_range']
        humid_min, humid_max = pattern['humidity_range']
        
        # Base temperature with seasonal and daily variation
        season_temp = (temp_max + temp_min) / 2
        if pattern['seasonal_pattern'] == 'coastal':
            season_temp += 3 * np.sin(2 * np.pi * (month - 1) / 12)
        elif pattern['seasonal_pattern'] == 'continental':
            season_temp += 15 * np.sin(2 * np.pi * (month - 1) / 12)
        else:
            season_temp += 8 * np.sin(2 * np.pi * (month - 1) / 12)
            
        # Add daily variation
        temp = season_temp + 5 * np.sin(2 * np.pi * (hour - 6) / 24)
        
        return {
            'temperature': np.clip(temp + np.random.normal(0, 1), temp_min, temp_max),
            'humidity': np.clip(
                (humid_max + humid_min) / 2 + (20 if is_monsoon else 0) + np.random.normal(0, 5),
                humid_min, humid_max
            ),
            'precipitation': (np.random.exponential(20) if is_monsoon and pattern['rainfall_intensity'] == 'very_high'
                            else np.random.exponential(15) if is_monsoon and pattern['rainfall_intensity'] == 'high'
                            else np.random.exponential(10) if is_monsoon
                            else 0),
            'solar_radiation': 1000 * np.sin(np.pi * hour / 24) * (0.3 if is_monsoon else 1),
            'cloud_cover': 80 if is_monsoon else 30 + np.random.normal(0, 10),
            'wind_speed': np.random.lognormal(2, 0.5),
            'wind_direction': np.random.uniform(0, 360)
        }
    
    def generate_energy_demand(self, timestamp: datetime, city: str, 
                             weather: Dict, previous_demand: Optional[float] = None) -> Dict:
        """Generate realistic energy demand based on historical patterns and weather"""
        pattern = self.energy_patterns[city]
        infrastructure = self.infrastructure[city]
        
        # Base load calculation with yearly growth
        years_from_base = timestamp.year - self.start_date.year
        base_load = pattern['base_load'] * (1 + 0.07) ** years_from_base  # 7% yearly growth
        
        # Time-based factors
        hour = timestamp.hour
        is_weekend = timestamp.weekday() >= 5
        is_holiday = timestamp.strftime('%Y-%m-%d') in INDIAN_HOLIDAYS
        
        # Peak hours definition
        morning_peak = hour in PEAK_HOURS['Morning']
        evening_peak = hour in PEAK_HOURS['Evening']
        
        # Demand multipliers
        time_multiplier = pattern['peak_multiplier'] if (morning_peak or evening_peak) else 1.0
        day_multiplier = 0.8 if (is_weekend or is_holiday) else 1.0
        
        # Weather impact
        temp_factor = 1.0
        if weather['temperature'] > 30:  # AC usage
            temp_factor += 0.05 * (weather['temperature'] - 30)
        elif weather['temperature'] < 15:  # Heating usage
            temp_factor += 0.03 * (15 - weather['temperature'])
        
        # Calculate total demand
        total_demand = (base_load * time_multiplier * day_multiplier * temp_factor * 
                       (1 + np.random.normal(0, 0.02)))
        
        # Apply infrastructure constraints
        total_demand = min(total_demand, infrastructure['grid_capacity'])
        
        return {
            'total_demand': total_demand,
            'residential_demand': total_demand * pattern['residential_ratio'],
            'commercial_demand': total_demand * pattern['commercial_ratio'],
            'industrial_demand': total_demand * pattern['industrial_ratio'],
            'peak_demand': total_demand if (morning_peak or evening_peak) else total_demand * 0.8,
            'base_demand': base_load
        }
    
    def generate_energy_supply(self, timestamp: datetime, city: str,
                             weather: Dict, demand: Dict) -> Dict:
        """Generate energy supply mix based on infrastructure and weather"""
        infrastructure = self.infrastructure[city]
        total_required = demand['total_demand']
        
        # Calculate renewable generation based on weather
        solar_potential = weather['solar_radiation'] / 1000.0 * (1 - weather['cloud_cover']/100)
        wind_potential = np.clip((weather['wind_speed'] - 3) / 10, 0, 1)
        
        # Calculate supply from different sources
        renewable_capacity = infrastructure['grid_capacity'] * infrastructure['renewable_share']
        solar_supply = renewable_capacity * 0.6 * solar_potential
        wind_supply = renewable_capacity * 0.4 * wind_potential
        
        # Thermal and other sources
        remaining_demand = total_required - (solar_supply + wind_supply)
        thermal_coal = remaining_demand * 0.7
        thermal_gas = remaining_demand * 0.2
        nuclear = remaining_demand * 0.1
        
        # Account for distribution losses
        loss_factor = 1 + infrastructure['distribution_loss']
        
        return {
            'total_supply': total_required * loss_factor,
            'solar_generation': solar_supply,
            'wind_generation': wind_supply,
            'thermal_coal': thermal_coal,
            'thermal_gas': thermal_gas,
            'nuclear': nuclear,
            'distribution_loss': total_required * infrastructure['distribution_loss']
        }
    
    def generate_dataset(self) -> pd.DataFrame:
        """Generate complete dataset for all cities"""
        data_points = []
        previous_demand = {city: None for city in self.cities}
        
        current_date = self.start_date
        total_hours = int((self.end_date - self.start_date).total_seconds() / 3600)
        processed_hours = 0
        
        while current_date <= self.end_date:
            for city in self.cities:
                # Generate features
                weather = self.generate_weather_features(current_date, city)
                demand = self.generate_energy_demand(current_date, city, weather, previous_demand[city])
                supply = self.generate_energy_supply(current_date, city, weather, demand)
                
                # Store previous demand
                previous_demand[city] = demand['total_demand']
                
                # Combine all features
                data_point = {
                    'timestamp': current_date,
                    'city': city,
                    **weather,
                    **demand,
                    **supply
                }
                
                data_points.append(data_point)
            
            current_date += timedelta(hours=1)
            processed_hours += 1
            
            if processed_hours % 1000 == 0:
                progress = (processed_hours / total_hours) * 100
                logger.info(f"Progress: {progress:.2f}% complete")
        
        # Convert to DataFrame
        df = pd.DataFrame(data_points)
        
        # Add derived features
        df = self._add_derived_features(df)
        
        return df
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features to the dataset"""
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = df['day_of_week'] >= 5
        
        # Calculate rolling means for each city
        for city in self.cities:
            city_mask = df['city'] == city
            df.loc[city_mask, 'demand_ma_24h'] = df.loc[city_mask, 'total_demand'].rolling(24).mean()
            df.loc[city_mask, 'demand_ma_7d'] = df.loc[city_mask, 'total_demand'].rolling(168).mean()
        
        # Energy efficiency metrics
        df['renewable_ratio'] = (df['solar_generation'] + df['wind_generation']) / df['total_supply']
        df['peak_ratio'] = df['peak_demand'] / df['base_demand']
        df['load_factor'] = df['total_demand'] / df['peak_demand']
        
        return df
