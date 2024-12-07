from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class City(Base):
    __tablename__ = 'cities'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True)
    country = Column(String(100), nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    
    weather_data = relationship('WeatherData', back_populates='city')
    
    def __repr__(self):
        return f"<City {self.name}, {self.country}>"

class WeatherData(Base):
    __tablename__ = 'weather_data'
    
    id = Column(Integer, primary_key=True)
    city_id = Column(Integer, ForeignKey('cities.id'), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    temperature = Column(Float)
    pressure = Column(Float)
    humidity = Column(Float)
    wind_speed = Column(Float)
    wind_direction = Column(Float)
    rain_3h = Column(Float)
    
    city = relationship('City', back_populates='weather_data')
    
    def __repr__(self):
        return f"<WeatherData {self.city.name} @ {self.timestamp}>"

class EnergyConsumption(Base):
    __tablename__ = 'energy_consumption'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    total_load = Column(Float, nullable=False)
    fossil_fuel = Column(Float)
    hydro = Column(Float)
    nuclear = Column(Float)
    solar = Column(Float)
    wind = Column(Float)
    
    def __repr__(self):
        return f"<EnergyConsumption @ {self.timestamp}>"

class Prediction(Base):
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    predicted_load = Column(Float, nullable=False)
    actual_load = Column(Float)
    model_name = Column(String(100), nullable=False)
    mae = Column(Float)
    mse = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<Prediction {self.model_name} @ {self.timestamp}>"

class ModelMetrics(Base):
    __tablename__ = 'model_metrics'
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String(100), nullable=False)
    training_date = Column(DateTime, nullable=False)
    mae = Column(Float, nullable=False)
    mse = Column(Float, nullable=False)
    r2_score = Column(Float, nullable=False)
    parameters = Column(String(500))  # JSON string of model parameters
    
    def __repr__(self):
        return f"<ModelMetrics {self.model_name} @ {self.training_date}>"
