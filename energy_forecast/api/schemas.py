"""API request and response schemas"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional
from datetime import datetime

class WeatherData(BaseModel):
    temperature: float = Field(..., ge=-50, le=50)
    humidity: float = Field(..., ge=0, le=100)
    cloud_cover: float = Field(..., ge=0, le=100)
    solar_radiation: float = Field(..., ge=0, le=1200)
    wind_speed: float = Field(..., ge=0, le=100)
    precipitation: float = Field(..., ge=0, le=200)

class ForecastRequest(BaseModel):
    city: str
    start_date: datetime
    end_date: datetime
    weather_forecast: Optional[List[WeatherData]]
    
    @validator('city')
    def validate_city(cls, v):
        valid_cities = ['Mumbai', 'Delhi']  # Add more cities as needed
        if v not in valid_cities:
            raise ValueError(f'City must be one of {valid_cities}')
        return v
    
    @validator('end_date')
    def validate_dates(cls, v, values):
        if 'start_date' in values and v <= values['start_date']:
            raise ValueError('end_date must be after start_date')
        return v

class ForecastResponse(BaseModel):
    city: str
    forecast_period: Dict[str, datetime]
    demand_forecast: Dict[str, float]
    energy_mix: Dict[str, Dict[str, float]]
    confidence_intervals: Dict[str, Dict[str, float]]
    production_schedule: Dict[str, List[Dict]]
    total_cost: float
    metadata: Dict

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str]
    timestamp: datetime

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime

class ValidationError(BaseModel):
    field: str
    message: str

class DataQualityResponse(BaseModel):
    status: str
    validation_errors: Optional[List[ValidationError]]
    quality_metrics: Dict[str, float]
    recommendations: Optional[List[str]]
