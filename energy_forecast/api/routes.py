"""API routes for energy forecasting"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from energy_forecast.api.schemas import (
    ForecastRequest,
    ForecastResponse,
    ErrorResponse,
    HealthResponse,
    DataQualityResponse
)
from energy_forecast.api.auth import get_current_user, verify_api_key
from energy_forecast.api.data_validation import DataValidator
from energy_forecast.core.models.energy_forecast_model import EnergyForecastModel

# Initialize routers
router = APIRouter()
health_router = APIRouter(prefix="/health", tags=["health"])
forecast_router = APIRouter(prefix="/forecast", tags=["forecast"])
data_router = APIRouter(prefix="/data", tags=["data"])

# Initialize components
data_validator = DataValidator()

@health_router.get("/", response_model=HealthResponse)
async def health_check():
    """Check API health"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.utcnow()
    }

@forecast_router.post("/demand", response_model=ForecastResponse)
async def forecast_demand(
    request: ForecastRequest,
    api_key: str = Depends(verify_api_key),
    current_user: str = Depends(get_current_user)
):
    """Generate energy demand forecast"""
    try:
        # Validate request data
        valid, errors = data_validator.validate_forecast_request(
            request.dict(),
            request.city
        )
        if not valid:
            raise HTTPException(
                status_code=400,
                detail={"validation_errors": errors}
            )
        
        # Initialize model
        model = EnergyForecastModel(request.city)
        
        # Generate forecast
        forecast = model.forecast_demand(
            current_data=request.weather_forecast,
            forecast_horizon=(request.end_date - request.start_date).days * 24
        )
        
        # Optimize energy mix
        energy_mix = model.optimize_energy_mix(
            forecast,
            request.weather_forecast
        )
        
        return {
            "city": request.city,
            "forecast_period": {
                "start": request.start_date,
                "end": request.end_date
            },
            "demand_forecast": forecast['demand_mean'].to_dict(),
            "energy_mix": energy_mix['energy_mix'].to_dict(),
            "confidence_intervals": {
                "lower": forecast['demand_lower'].to_dict(),
                "upper": forecast['demand_upper'].to_dict()
            },
            "production_schedule": energy_mix['schedule'].to_dict('records'),
            "total_cost": float(energy_mix['total_cost'].sum()),
            "metadata": {
                "generated_at": datetime.utcnow(),
                "model_version": "1.0.0",
                "user": current_user
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@forecast_router.get("/cities", response_model=List[str])
async def list_cities(
    api_key: str = Depends(verify_api_key)
):
    """List available cities"""
    return ["Mumbai", "Delhi"]  # Add more cities as needed

@data_router.post("/validate", response_model=DataQualityResponse)
async def validate_data(
    data: Dict,
    city: str = Query(..., description="City name"),
    api_key: str = Depends(verify_api_key)
):
    """Validate data quality"""
    try:
        # Validate data
        valid, errors = data_validator.validate_forecast_request(data, city)
        
        # Calculate quality metrics
        quality_metrics = data_validator.check_data_quality(data)
        
        # Generate recommendations
        recommendations = []
        if quality_metrics.get('completeness', 1) < 0.95:
            recommendations.append("Consider filling missing values")
        if quality_metrics.get('validity', 1) < 0.9:
            recommendations.append("Check data ranges and outliers")
        
        return {
            "status": "valid" if valid else "invalid",
            "validation_errors": [
                {"field": err.split(":")[0], "message": err.split(":")[1]}
                for err in errors
            ] if errors else None,
            "quality_metrics": quality_metrics,
            "recommendations": recommendations if recommendations else None
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
