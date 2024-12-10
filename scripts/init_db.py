import os
import sys
import logging
from datetime import datetime, timedelta
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.models.database import SessionLocal, init_db
from backend.models.predictions import City, ConsumptionData, Forecast

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def seed_cities(db):
    """Seed initial city data"""
    cities = [
        {
            "name": "Mumbai",
            "region": "Maharashtra",
            "population": 20185064,
            "climate_zone": "Tropical"
        },
        {
            "name": "Delhi",
            "region": "Delhi",
            "population": 32941000,
            "climate_zone": "Semi-arid"
        },
        {
            "name": "Bangalore",
            "region": "Karnataka",
            "population": 12425304,
            "climate_zone": "Tropical Savanna"
        },
        {
            "name": "Chennai",
            "region": "Tamil Nadu",
            "population": 11503293,
            "climate_zone": "Tropical Wet"
        },
        {
            "name": "Hyderabad",
            "region": "Telangana",
            "population": 10268653,
            "climate_zone": "Hot Semi-arid"
        }
    ]
    
    city_objects = []
    for city_data in cities:
        city = City(**city_data)
        db.add(city)
        city_objects.append(city)
    
    db.commit()
    logger.info("Cities seeded successfully")
    return city_objects

def seed_consumption_data(db, cities):
    """Seed historical consumption data"""
    now = datetime.now()
    
    for city in cities:
        # Generate 30 days of hourly data
        for day in range(30):
            for hour in range(24):
                timestamp = now - timedelta(days=day, hours=hour)
                
                # Create realistic consumption pattern
                base_consumption = 2000  # Base load
                time_factor = 1 + (hour / 24)  # Daily pattern
                seasonal_factor = 1 + (0.2 * (day % 7) / 7)  # Weekly pattern
                
                consumption = ConsumptionData(
                    city_id=city.id,
                    timestamp=timestamp,
                    consumption=base_consumption * time_factor * seasonal_factor,
                    temperature=25 + (hour * 0.5) + (day * 0.1),
                    humidity=60 + (hour * 0.8) - (day * 0.2)
                )
                db.add(consumption)
        
        logger.info(f"Consumption data seeded for {city.name}")
        db.commit()

def seed_forecasts(db, cities):
    """Seed forecast data"""
    now = datetime.now()
    
    for city in cities:
        # Generate 7 days of hourly forecasts
        for day in range(7):
            for hour in range(24):
                timestamp = now + timedelta(days=day, hours=hour)
                
                # Create forecast with decreasing confidence over time
                confidence = 0.95 - (day * 0.05) - (hour * 0.002)
                base_prediction = 2500 + (hour * 100) + (day * 50)
                
                forecast = Forecast(
                    city_id=city.id,
                    timestamp=timestamp,
                    predicted_consumption=base_prediction,
                    confidence_level=max(0.6, confidence)
                )
                db.add(forecast)
        
        logger.info(f"Forecast data seeded for {city.name}")
        db.commit()

def main():
    """Main initialization function"""
    load_dotenv()
    
    try:
        # Initialize database
        init_db()
        logger.info("Database initialized successfully")
        
        # Get database session
        db = SessionLocal()
        
        try:
            # Seed data
            cities = seed_cities(db)
            seed_consumption_data(db, cities)
            seed_forecasts(db, cities)
            
            logger.info("All data seeded successfully")
            
        except SQLAlchemyError as e:
            logger.error(f"Database error during seeding: {e}")
            db.rollback()
            raise
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error during database initialization: {e}")
        raise

if __name__ == "__main__":
    main()
