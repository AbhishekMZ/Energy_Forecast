from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from backend.models import predictions, database, schemas
from backend.ml_models.forecast import EnergyForecaster

app = FastAPI(title="Energy Forecast API")

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database dependency
def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Initialize ML model
forecaster = EnergyForecaster()

@app.get("/api/cities", response_model=List[str])
async def get_cities(db: Session = Depends(get_db)):
    """Get list of available cities"""
    return ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad"]

@app.get("/api/current-usage/{city}")
async def get_current_usage(city: str, db: Session = Depends(get_db)):
    """Get current energy usage for a city"""
    try:
        # Get latest reading from database
        latest_usage = predictions.get_latest_usage(db, city)
        return {
            "city": city,
            "current_usage": latest_usage,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Data not found for {city}")

@app.get("/api/forecast/{city}")
async def get_forecast(
    city: str,
    start_date: str,
    end_date: str,
    db: Session = Depends(get_db)
):
    """Get energy consumption forecast for a city"""
    try:
        # Convert dates
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
        
        # Get historical data
        historical_data = predictions.get_historical_data(db, city, start, end)
        
        # Generate forecast
        forecast_data = forecaster.predict(historical_data)
        
        return {
            "city": city,
            "forecast": forecast_data,
            "start_date": start_date,
            "end_date": end_date
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/api/efficiency/{city}")
async def get_efficiency(city: str, db: Session = Depends(get_db)):
    """Get energy efficiency metrics for a city"""
    try:
        efficiency_data = predictions.get_efficiency_metrics(db, city)
        return {
            "city": city,
            "efficiency": efficiency_data["efficiency"],
            "peak_hours": efficiency_data["peak_hours"],
            "off_peak_hours": efficiency_data["off_peak_hours"]
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/api/recommendations/{city}")
async def get_recommendations(city: str, db: Session = Depends(get_db)):
    """Get energy optimization recommendations for a city"""
    try:
        current_patterns = predictions.get_consumption_patterns(db, city)
        recommendations = forecaster.generate_recommendations(current_patterns)
        return {
            "city": city,
            "recommendations": recommendations
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/api/stats/{city}")
async def get_stats(city: str, db: Session = Depends(get_db)):
    """Get comprehensive stats for a city"""
    try:
        stats = predictions.get_city_stats(db, city)
        return {
            "city": city,
            "current_usage": stats["current_usage"],
            "predicted_peak": stats["predicted_peak"],
            "efficiency": stats["efficiency"],
            "accuracy": stats["accuracy"]
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
