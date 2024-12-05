from typing import Dict, List, Optional
from pydantic import BaseModel
from datetime import datetime

class WeatherFeatures(BaseModel):
    temperature: float            # in Celsius
    humidity: float              # percentage
    precipitation: float         # in mm
    cloud_cover: float          # percentage
    wind_speed: float           # in km/h
    wind_direction: float       # in degrees
    air_pressure: float         # in hPa
    visibility: float           # in km
    solar_radiation: float      # in W/m²

class EnergyDemand(BaseModel):
    total_demand: float         # in MW
    residential_demand: float   # in MW
    commercial_demand: float    # in MW
    industrial_demand: float    # in MW
    peak_demand: float         # in MW
    base_demand: float         # in MW

class EnergySupply(BaseModel):
    total_supply: float        # in MW
    solar_generation: float    # in MW
    wind_generation: float     # in MW
    hydro_generation: float    # in MW
    thermal_coal: float        # in MW
    thermal_gas: float         # in MW
    nuclear: float            # in MW
    biomass: float           # in MW
    other_renewable: float   # in MW

class CityFeatures(BaseModel):
    population: int
    area: float              # in km²
    industrial_zones: int    # number of major industrial zones
    commercial_centers: int  # number of major commercial centers
    gdp_per_capita: float   # in INR
    power_infrastructure: Dict[str, int]  # substations, transformers etc.
    grid_reliability: float  # percentage uptime
    electrification_rate: float  # percentage of area covered

class DerivedFeatures(BaseModel):
    is_weekend: bool
    is_holiday: bool
    season: str             # Summer, Monsoon, Winter, Spring
    hour_of_day: int
    day_of_week: int
    month: int
    is_peak_hour: bool
    cooling_degree_days: float
    heating_degree_days: float
    previous_day_demand: float
    previous_week_demand: float
    demand_growth_rate: float

class EnergyDataPoint(BaseModel):
    timestamp: datetime
    city: str
    weather: WeatherFeatures
    demand: EnergyDemand
    supply: EnergySupply
    city_features: CityFeatures
    derived_features: DerivedFeatures
    
    class Config:
        arbitrary_types_allowed = True
