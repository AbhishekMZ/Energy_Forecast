import logging
from datetime import datetime, timedelta
from sqlalchemy.exc import SQLAlchemyError
from database import init_db, SessionLocal
from predictions import City, ConsumptionData, Forecast

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def seed_initial_data():
    """Seed initial data for testing and development"""
    try:
        db = SessionLocal()
        
        # Create sample cities
        cities = [
            City(name="Mumbai", region="Maharashtra", population=20185064, climate_zone="Tropical"),
            City(name="Delhi", region="Delhi", population=32941000, climate_zone="Semi-arid"),
            City(name="Bangalore", region="Karnataka", population=12425304, climate_zone="Tropical Savanna"),
            City(name="Chennai", region="Tamil Nadu", population=11503293, climate_zone="Tropical Wet"),
            City(name="Hyderabad", region="Telangana", population=10268653, climate_zone="Hot Semi-arid")
        ]
        
        db.add_all(cities)
        db.commit()
        logger.info("Sample cities added successfully")

        # Create sample consumption data
        now = datetime.now()
        for city in cities:
            # Generate 7 days of hourly data
            for day in range(7):
                for hour in range(24):
                    timestamp = now - timedelta(days=day, hours=hour)
                    consumption = ConsumptionData(
                        city_id=city.id,
                        timestamp=timestamp,
                        consumption=2000 + (hour * 100) + (day * 50),  # Sample pattern
                        temperature=25 + (hour * 0.5),  # Sample temperature pattern
                        humidity=60 + (hour * 1)  # Sample humidity pattern
                    )
                    db.add(consumption)
            
            # Generate sample forecasts
            for hour in range(24):
                timestamp = now + timedelta(hours=hour)
                forecast = Forecast(
                    city_id=city.id,
                    timestamp=timestamp,
                    predicted_consumption=2500 + (hour * 100),
                    confidence_level=0.85 + (hour * 0.005)
                )
                db.add(forecast)
        
        db.commit()
        logger.info("Sample consumption data and forecasts added successfully")

    except SQLAlchemyError as e:
        logger.error(f"Database error during seeding: {e}")
        db.rollback()
        raise
    except Exception as e:
        logger.error(f"Error during data seeding: {e}")
        db.rollback()
        raise
    finally:
        db.close()

def main():
    """Main initialization function"""
    try:
        # Initialize database tables
        init_db()
        logger.info("Database initialized successfully")

        # Seed initial data
        seed_initial_data()
        logger.info("Initial data seeded successfully")

    except Exception as e:
        logger.error(f"Error during database initialization: {e}")
        raise

if __name__ == "__main__":
    main()
