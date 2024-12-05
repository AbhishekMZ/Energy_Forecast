import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from typing import Dict, List, Optional
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HistoricalDataLoader:
    def __init__(self):
        self.weather_api_key = None  # You'll need to set this
        self.base_path = "data/historical"
        
    def fetch_historical_weather(self, city: str, start_year: int = 2018, end_year: int = 2023):
        """
        Fetch historical weather data for Indian cities
        Using Visual Crossing Weather API as an example
        """
        city_coordinates = {
            "Mumbai": {"lat": 19.0760, "lon": 72.8777},
            "Delhi": {"lat": 28.6139, "lon": 77.2090},
            "Bangalore": {"lat": 12.9716, "lon": 77.5946},
            "Chennai": {"lat": 13.0827, "lon": 80.2707},
            "Kolkata": {"lat": 22.5726, "lon": 88.3639},
            "Hyderabad": {"lat": 17.3850, "lon": 78.4867},
            "Pune": {"lat": 18.5204, "lon": 73.8567}
        }
        
        if city not in city_coordinates:
            raise ValueError(f"City {city} not supported")
            
        coords = city_coordinates[city]
        
        # Structure to store historical data
        historical_data = []
        
        try:
            # Example using Visual Crossing Weather API
            for year in range(start_year, end_year + 1):
                url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{coords['lat']},{coords['lon']}/{year}-01-01/{year}-12-31"
                params = {
                    'unitGroup': 'metric',
                    'key': self.weather_api_key,
                    'include': 'hours'
                }
                
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    historical_data.extend(self._process_weather_data(data, city))
                    
            return pd.DataFrame(historical_data)
            
        except Exception as e:
            logger.error(f"Error fetching weather data for {city}: {str(e)}")
            return self._generate_synthetic_historical_data(city, start_year, end_year)
    
    def _generate_synthetic_historical_data(self, city: str, start_year: int, end_year: int):
        """
        Generate synthetic but realistic historical weather data based on known patterns
        """
        city_weather_patterns = {
            "Mumbai": {
                "temp_range": (22, 35),
                "humidity_range": (60, 90),
                "rainfall_months": [6, 7, 8, 9],  # June to September
                "rainfall_intensity": "high",
                "seasonal_pattern": "coastal"
            },
            "Delhi": {
                "temp_range": (8, 45),
                "humidity_range": (30, 80),
                "rainfall_months": [7, 8, 9],
                "rainfall_intensity": "moderate",
                "seasonal_pattern": "continental"
            },
            "Bangalore": {
                "temp_range": (15, 35),
                "humidity_range": (40, 85),
                "rainfall_months": [5, 6, 9, 10],
                "rainfall_intensity": "moderate",
                "seasonal_pattern": "plateau"
            },
            "Chennai": {
                "temp_range": (24, 38),
                "humidity_range": (65, 85),
                "rainfall_months": [10, 11, 12],
                "rainfall_intensity": "very_high",
                "seasonal_pattern": "coastal"
            },
            "Kolkata": {
                "temp_range": (15, 40),
                "humidity_range": (50, 85),
                "rainfall_months": [6, 7, 8, 9],
                "rainfall_intensity": "high",
                "seasonal_pattern": "subtropical"
            }
        }
        
        pattern = city_weather_patterns[city]
        data = []
        
        current_date = datetime(start_year, 1, 1)
        end_date = datetime(end_year, 12, 31, 23, 59, 59)
        
        while current_date <= end_date:
            # Temperature generation with seasonal variation
            month = current_date.month
            hour = current_date.hour
            
            # Base temperature calculation
            if pattern["seasonal_pattern"] == "coastal":
                base_temp = np.mean(pattern["temp_range"]) + \
                          3 * np.sin(2 * np.pi * (month - 1) / 12)
            elif pattern["seasonal_pattern"] == "continental":
                base_temp = np.mean(pattern["temp_range"]) + \
                          15 * np.sin(2 * np.pi * (month - 1) / 12)
            else:
                base_temp = np.mean(pattern["temp_range"]) + \
                          8 * np.sin(2 * np.pi * (month - 1) / 12)
            
            # Add daily variation
            temp = base_temp + 5 * np.sin(2 * np.pi * (hour - 6) / 24)
            
            # Add some random variation
            temp += np.random.normal(0, 1)
            
            # Humidity calculation
            base_humidity = np.mean(pattern["humidity_range"])
            if month in pattern["rainfall_months"]:
                humidity = base_humidity + 15
            else:
                humidity = base_humidity
            
            # Add some random variation to humidity
            humidity += np.random.normal(0, 5)
            humidity = np.clip(humidity, 0, 100)
            
            # Rainfall calculation
            rainfall = 0
            if month in pattern["rainfall_months"]:
                if pattern["rainfall_intensity"] == "very_high":
                    rainfall = np.random.exponential(20) if np.random.random() < 0.4 else 0
                elif pattern["rainfall_intensity"] == "high":
                    rainfall = np.random.exponential(15) if np.random.random() < 0.3 else 0
                else:
                    rainfall = np.random.exponential(10) if np.random.random() < 0.2 else 0
            
            # Solar radiation calculation (simplified)
            solar_rad = 1000 * np.sin(np.pi * hour / 24) * (1 - 0.75 * (rainfall > 0))
            
            data.append({
                'timestamp': current_date,
                'city': city,
                'temperature': temp,
                'humidity': humidity,
                'precipitation': rainfall,
                'solar_radiation': max(0, solar_rad),
                'cloud_cover': 100 if rainfall > 0 else np.random.uniform(0, 60),
                'wind_speed': np.random.lognormal(2, 0.5),
                'wind_direction': np.random.uniform(0, 360)
            })
            
            current_date += timedelta(hours=1)
            
        return pd.DataFrame(data)
    
    def load_energy_consumption_patterns(self, city: str):
        """
        Load or generate typical energy consumption patterns for each city
        """
        city_patterns = {
            "Mumbai": {
                "base_load": 3000,  # MW
                "peak_multiplier": 1.6,
                "industrial_share": 0.45,
                "commercial_share": 0.35,
                "residential_share": 0.20,
                "seasonal_variation": 0.25,  # 25% variation between seasons
                "yearly_growth": 0.08  # 8% yearly growth
            },
            "Delhi": {
                "base_load": 4500,
                "peak_multiplier": 1.8,
                "industrial_share": 0.35,
                "commercial_share": 0.40,
                "residential_share": 0.25,
                "seasonal_variation": 0.35,
                "yearly_growth": 0.07
            },
            "Bangalore": {
                "base_load": 2500,
                "peak_multiplier": 1.5,
                "industrial_share": 0.50,
                "commercial_share": 0.35,
                "residential_share": 0.15,
                "seasonal_variation": 0.15,
                "yearly_growth": 0.09
            },
            "Chennai": {
                "base_load": 2800,
                "peak_multiplier": 1.6,
                "industrial_share": 0.40,
                "commercial_share": 0.35,
                "residential_share": 0.25,
                "seasonal_variation": 0.20,
                "yearly_growth": 0.075
            },
            "Kolkata": {
                "base_load": 2200,
                "peak_multiplier": 1.5,
                "industrial_share": 0.40,
                "commercial_share": 0.30,
                "residential_share": 0.30,
                "seasonal_variation": 0.30,
                "yearly_growth": 0.065
            }
        }
        
        return city_patterns.get(city, city_patterns["Mumbai"])  # Default to Mumbai if city not found
    
    def get_industrial_calendar(self, city: str, year: int):
        """
        Generate industrial working calendar including holidays and maintenance periods
        """
        holidays = {
            f"{year}-01-26": "Republic Day",
            f"{year}-08-15": "Independence Day",
            f"{year}-10-02": "Gandhi Jayanti",
            f"{year}-05-01": "Labor Day",
            # Add more holidays as needed
        }
        
        # Add city-specific industrial shutdown periods
        if city == "Mumbai":
            holidays.update({
                f"{year}-09-{day:02d}": "Ganesh Chaturthi"
                for day in range(10, 20)
            })
        elif city == "Delhi":
            # Add Diwali period
            holidays.update({
                f"{year}-11-{day:02d}": "Diwali Period"
                for day in range(1, 6)
            })
            
        return holidays
    
    def get_city_infrastructure(self, city: str):
        """
        Return infrastructure details for each city
        """
        return {
            "Mumbai": {
                "grid_capacity": 4000,  # MW
                "renewable_share": 0.10,
                "distribution_loss": 0.15,
                "backup_capacity": 1000,
                "smart_grid_coverage": 0.60
            },
            "Delhi": {
                "grid_capacity": 5500,
                "renewable_share": 0.15,
                "distribution_loss": 0.18,
                "backup_capacity": 1500,
                "smart_grid_coverage": 0.70
            },
            "Bangalore": {
                "grid_capacity": 3000,
                "renewable_share": 0.25,
                "distribution_loss": 0.12,
                "backup_capacity": 800,
                "smart_grid_coverage": 0.80
            },
            "Chennai": {
                "grid_capacity": 3500,
                "renewable_share": 0.20,
                "distribution_loss": 0.14,
                "backup_capacity": 900,
                "smart_grid_coverage": 0.65
            },
            "Kolkata": {
                "grid_capacity": 2800,
                "renewable_share": 0.12,
                "distribution_loss": 0.16,
                "backup_capacity": 700,
                "smart_grid_coverage": 0.55
            }
        }.get(city, None)
