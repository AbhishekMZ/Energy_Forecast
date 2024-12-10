from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from .database import Base
from datetime import datetime

class City(Base):
    __tablename__ = "cities"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    region = Column(String)
    population = Column(Integer)
    climate_zone = Column(String)

    # Relationships
    consumption_data = relationship("ConsumptionData", back_populates="city")
    forecasts = relationship("Forecast", back_populates="city")

class ConsumptionData(Base):
    __tablename__ = "consumption_data"

    id = Column(Integer, primary_key=True, index=True)
    city_id = Column(Integer, ForeignKey("cities.id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    consumption = Column(Float)
    temperature = Column(Float)
    humidity = Column(Float)
    
    # Relationships
    city = relationship("City", back_populates="consumption_data")

class Forecast(Base):
    __tablename__ = "forecasts"

    id = Column(Integer, primary_key=True, index=True)
    city_id = Column(Integer, ForeignKey("cities.id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    predicted_consumption = Column(Float)
    confidence_level = Column(Float)
    
    # Relationships
    city = relationship("City", back_populates="forecasts")

# Database operations
def get_latest_usage(db, city_name: str) -> float:
    """Get the latest energy usage for a city"""
    city = db.query(City).filter(City.name == city_name).first()
    if not city:
        return 0.0
    
    latest_data = (db.query(ConsumptionData)
                  .filter(ConsumptionData.city_id == city.id)
                  .order_by(ConsumptionData.timestamp.desc())
                  .first())
    
    return latest_data.consumption if latest_data else 0.0

def get_historical_data(db, city_name: str, start_date: datetime, end_date: datetime):
    """Get historical consumption data for a city"""
    city = db.query(City).filter(City.name == city_name).first()
    if not city:
        return []
    
    historical_data = (db.query(ConsumptionData)
                      .filter(ConsumptionData.city_id == city.id)
                      .filter(ConsumptionData.timestamp.between(start_date, end_date))
                      .order_by(ConsumptionData.timestamp)
                      .all())
    
    return [
        {
            "timestamp": data.timestamp,
            "consumption": data.consumption,
            "temperature": data.temperature,
            "humidity": data.humidity
        }
        for data in historical_data
    ]

def get_efficiency_metrics(db, city_name: str):
    """Calculate efficiency metrics for a city"""
    city = db.query(City).filter(City.name == city_name).first()
    if not city:
        return {"efficiency": 0, "peak_hours": [], "off_peak_hours": []}
    
    # Calculate efficiency based on historical data
    consumption_data = (db.query(ConsumptionData)
                       .filter(ConsumptionData.city_id == city.id)
                       .order_by(ConsumptionData.timestamp.desc())
                       .limit(24)
                       .all())
    
    if not consumption_data:
        return {"efficiency": 0, "peak_hours": [], "off_peak_hours": []}
    
    # Simple efficiency calculation
    total_consumption = sum(data.consumption for data in consumption_data)
    peak_consumption = max(data.consumption for data in consumption_data)
    efficiency = 1 - (peak_consumption / (total_consumption / len(consumption_data)))
    
    return {
        "efficiency": round(efficiency * 100, 2),
        "peak_hours": ["10:00", "14:00"],
        "off_peak_hours": ["23:00", "05:00"]
    }

def get_consumption_patterns(db, city_name: str):
    """Get consumption patterns for a city"""
    city = db.query(City).filter(City.name == city_name).first()
    if not city:
        return []
    
    # Get last 7 days of data
    patterns = (db.query(ConsumptionData)
               .filter(ConsumptionData.city_id == city.id)
               .order_by(ConsumptionData.timestamp.desc())
               .limit(168)  # 24 * 7
               .all())
    
    return [
        {
            "timestamp": data.timestamp,
            "consumption": data.consumption
        }
        for data in patterns
    ]

def get_city_stats(db, city_name: str):
    """Get comprehensive stats for a city"""
    city = db.query(City).filter(City.name == city_name).first()
    if not city:
        return {
            "current_usage": 0,
            "predicted_peak": 0,
            "efficiency": 0,
            "accuracy": 0
        }
    
    # Get latest consumption
    current_usage = get_latest_usage(db, city_name)
    
    # Get latest forecast
    latest_forecast = (db.query(Forecast)
                      .filter(Forecast.city_id == city.id)
                      .order_by(Forecast.timestamp.desc())
                      .first())
    
    # Get efficiency metrics
    efficiency_data = get_efficiency_metrics(db, city_name)
    
    return {
        "current_usage": current_usage,
        "predicted_peak": latest_forecast.predicted_consumption if latest_forecast else 0,
        "efficiency": efficiency_data["efficiency"],
        "accuracy": latest_forecast.confidence_level if latest_forecast else 0
    }
