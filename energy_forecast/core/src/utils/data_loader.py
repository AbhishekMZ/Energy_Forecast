import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from typing import Dict, List
import logging
from ..config.constants import WEATHER_PATTERNS, INDIAN_HOLIDAYS

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
        historical_data = []
        
        try:
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
        """Generate synthetic but realistic historical weather data"""
        pattern = WEATHER_PATTERNS[city]
        data = []
        
        current_date = datetime(start_year, 1, 1)
        end_date = datetime(end_year, 12, 31, 23, 59, 59)
        
        while current_date <= end_date:
            month = current_date.month
            hour = current_date.hour
            
            # Temperature calculation
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
            temp += np.random.normal(0, 1)
            
            # Humidity calculation
            base_humidity = np.mean(pattern["humidity_range"])
            if month in pattern["rainfall_months"]:
                humidity = base_humidity + 15
            else:
                humidity = base_humidity
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
            
            # Solar radiation calculation
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
    
    def _process_weather_data(self, raw_data: Dict, city: str) -> List[Dict]:
        """Process raw weather API data into our format"""
        processed_data = []
        
        for day in raw_data['days']:
            date = datetime.strptime(day['datetime'], '%Y-%m-%d')
            
            for hour in day.get('hours', []):
                timestamp = date.replace(hour=int(hour['datetime'].split(':')[0]))
                
                processed_data.append({
                    'timestamp': timestamp,
                    'city': city,
                    'temperature': hour.get('temp', 0),
                    'humidity': hour.get('humidity', 0),
                    'precipitation': hour.get('precip', 0),
                    'cloud_cover': hour.get('cloudcover', 0),
                    'wind_speed': hour.get('windspeed', 0),
                    'wind_direction': hour.get('winddir', 0),
                    'solar_radiation': hour.get('solarradiation', 0)
                })
        
        return processed_data
    
    def get_industrial_calendar(self, city: str, year: int):
        """Generate industrial working calendar including holidays"""
        holidays = {f"{year}-{date}": name for date, name in INDIAN_HOLIDAYS.items()}
        
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
