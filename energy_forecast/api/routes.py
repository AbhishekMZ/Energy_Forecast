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
from .caching import CacheManager
from .batch_processor import ModelBatcher, EndpointBatcher
from ..core.utils.performance_monitoring import profile_endpoint

# Initialize routers
router = APIRouter()
health_router = APIRouter(prefix="/health", tags=["health"])
forecast_router = APIRouter(prefix="/forecast", tags=["forecast"])
data_router = APIRouter(prefix="/data", tags=["data"])

# Initialize components
data_validator = DataValidator()
cache_manager = CacheManager(settings.REDIS_URL)
model_batcher = ModelBatcher("energy_forecast", optimal_batch_size=32)
endpoint_batcher = EndpointBatcher("forecast_endpoint", max_batch_size=50)

@health_router.get("/", response_model=HealthResponse)
async def health_check():
    """Check API health"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.utcnow()
    }

@forecast_router.post("/demand")
@profile_endpoint("forecast_demand")
@cache_manager.cache_response("demand_forecast", ttl=3600)
async def forecast_demand(
    request: ForecastRequest,
    api_key: str = Depends(verify_api_key),
    current_user: str = Depends(get_current_user)
):
    """Generate energy demand forecast"""
    try:
        result = await endpoint_batcher.process_endpoint_request(
            request.dict(),
            _process_forecast_batch
        )
        return result if result is not None else await _process_single_forecast(request)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

async def _process_forecast_batch(requests: List[Dict]):
    """Process a batch of forecast requests."""
    try:
        # Prepare batch inputs
        batch_inputs = [
            {
                "city": req["city"],
                "weather": req["weather_forecast"],
                "start_date": req["start_date"],
                "end_date": req["end_date"]
            }
            for req in requests
        ]
        
        # Get predictions using model batcher
        predictions = await model_batcher.predict(
            batch_inputs,
            EnergyForecastModel.predict_batch
        )
        
        # Format responses
        return [
            {
                "city": req["city"],
                "forecast_period": {
                    "start": req["start_date"],
                    "end": req["end_date"]
                },
                "demand_forecast": pred["demand_mean"].to_dict(),
                "energy_mix": pred["energy_mix"].to_dict(),
                "confidence_intervals": {
                    "lower": pred["demand_lower"].to_dict(),
                    "upper": pred["demand_upper"].to_dict()
                },
                "production_schedule": pred["schedule"].to_dict('records'),
                "total_cost": float(pred["total_cost"].sum()),
                "metadata": {
                    "generated_at": datetime.utcnow(),
                    "model_version": "1.0.0",
                    "user": current_user
                }
            }
            for req, pred in zip(requests, predictions)
        ]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

async def _process_single_forecast(request: ForecastRequest):
    """Process a single forecast request."""
    input_data = {
        "city": request.city,
        "weather": request.weather_forecast,
        "start_date": request.start_date,
        "end_date": request.end_date
    }
    
    prediction = await model_batcher.predict(
        [input_data],
        EnergyForecastModel.predict_batch
    )
    
    return {
        "city": request.city,
        "forecast_period": {
            "start": request.start_date,
            "end": request.end_date
        },
        "demand_forecast": prediction[0]["demand_mean"].to_dict(),
        "energy_mix": prediction[0]["energy_mix"].to_dict(),
        "confidence_intervals": {
            "lower": prediction[0]["demand_lower"].to_dict(),
            "upper": prediction[0]["demand_upper"].to_dict()
        },
        "production_schedule": prediction[0]["schedule"].to_dict('records'),
        "total_cost": float(prediction[0]["total_cost"].sum()),
        "metadata": {
            "generated_at": datetime.utcnow(),
            "model_version": "1.0.0",
            "user": current_user
        }
    }

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

@router.get("/metrics/performance")
async def get_performance_metrics():
    """Get system performance metrics."""
    return {
        "model_performance": model_batcher.get_batch_stats(),
        "endpoint_performance": endpoint_batcher.get_endpoint_stats(),
        "cache_performance": cache_manager.profiler.get_stats()
    }
