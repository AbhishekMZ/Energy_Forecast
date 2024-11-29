from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime, timedelta
import pandas as pd
import logging
from typing import Optional, List, Dict, Any, Union
from ..utils.error_handling import (
    DatabaseError, error_handler, log_error,
    validate_dataframe
)

from .models import Base, City, WeatherData, EnergyConsumption, Prediction, ModelMetrics
from config import Config

# Setup logging
logging.basicConfig(level=Config.LOG_LEVEL)
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self.engine = create_engine(Config.SQLALCHEMY_DATABASE_URI)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def create_tables(self):
        """Create all database tables"""
        Base.metadata.create_all(self.engine)

    def add_city(self, name, country, latitude, longitude):
        """Add a new city to the database"""
        try:
            city = City(name=name, country=country, latitude=latitude, longitude=longitude)
            self.session.add(city)
            self.session.commit()
            return city
        except SQLAlchemyError as e:
            logger.error(f"Error adding city: {e}")
            self.session.rollback()
            return None

    def add_weather_data(self, data):
        """Add weather data for a city"""
        try:
            weather_data = WeatherData(
                city_id=data['city_id'],
                timestamp=data['timestamp'],
                temperature=data.get('temperature'),
                pressure=data.get('pressure'),
                humidity=data.get('humidity'),
                wind_speed=data.get('wind_speed'),
                wind_direction=data.get('wind_direction'),
                rain_3h=data.get('rain_3h')
            )
            self.session.add(weather_data)
            self.session.commit()
            return weather_data
        except SQLAlchemyError as e:
            logger.error(f"Error adding weather data: {e}")
            self.session.rollback()
            return None

    def add_energy_consumption(self, data):
        """Add energy consumption data"""
        try:
            consumption = EnergyConsumption(
                timestamp=data['timestamp'],
                total_load=data['total_load'],
                fossil_fuel=data.get('fossil_fuel'),
                hydro=data.get('hydro'),
                nuclear=data.get('nuclear'),
                solar=data.get('solar'),
                wind=data.get('wind')
            )
            self.session.add(consumption)
            self.session.commit()
            return consumption
        except SQLAlchemyError as e:
            logger.error(f"Error adding energy consumption: {e}")
            self.session.rollback()
            return None

    def add_prediction(self, data):
        """Add a new prediction"""
        try:
            prediction = Prediction(
                timestamp=data['timestamp'],
                predicted_load=data['predicted_load'],
                actual_load=data.get('actual_load'),
                model_name=data['model_name'],
                mae=data.get('mae'),
                mse=data.get('mse')
            )
            self.session.add(prediction)
            self.session.commit()
            return prediction
        except SQLAlchemyError as e:
            logger.error(f"Error adding prediction: {e}")
            self.session.rollback()
            return None

    def get_weather_data(self, city_id, start_date, end_date):
        """Get weather data for a city within a date range"""
        try:
            return self.session.query(WeatherData).filter(
                WeatherData.city_id == city_id,
                WeatherData.timestamp.between(start_date, end_date)
            ).all()
        except SQLAlchemyError as e:
            logger.error(f"Error retrieving weather data: {e}")
            return []

    def get_energy_consumption(self, start_date, end_date):
        """Get energy consumption data within a date range"""
        try:
            return self.session.query(EnergyConsumption).filter(
                EnergyConsumption.timestamp.between(start_date, end_date)
            ).all()
        except SQLAlchemyError as e:
            logger.error(f"Error retrieving energy consumption: {e}")
            return []

    def get_model_performance(self, model_name):
        """Get performance metrics for a specific model"""
        try:
            return self.session.query(ModelMetrics).filter(
                ModelMetrics.model_name == model_name
            ).order_by(ModelMetrics.training_date.desc()).first()
        except SQLAlchemyError as e:
            logger.error(f"Error retrieving model metrics: {e}")
            return None

    def get_recent_predictions(self, hours=24):
        """Get predictions from the last n hours"""
        try:
            start_time = datetime.utcnow() - timedelta(hours=hours)
            return self.session.query(Prediction).filter(
                Prediction.timestamp >= start_time
            ).order_by(Prediction.timestamp.desc()).all()
        except SQLAlchemyError as e:
            logger.error(f"Error retrieving recent predictions: {e}")
            return []

    def export_to_csv(self, query_results, filename):
        """Export query results to CSV file"""
        try:
            df = pd.DataFrame([vars(item) for item in query_results])
            df.to_csv(filename, index=False)
            return True
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return False

    def close(self):
        """Close the database session"""
        self.session.close()

class DatabaseOperations:
    def __init__(self, session):
        self.session = session
        self.logger = logging.getLogger(__name__)

    @error_handler
    def add_weather_data(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> None:
        """Add weather data to database with validation"""
        try:
            if isinstance(data, pd.DataFrame):
                validate_dataframe(
                    data,
                    required_columns=['timestamp', 'temperature', 'humidity'],
                    numeric_columns=['temperature', 'humidity'],
                    datetime_columns=['timestamp']
                )
                weather_records = data.to_dict('records')
            else:
                weather_records = [data]

            for record in weather_records:
                weather_entry = WeatherData(**record)
                self.session.add(weather_entry)
            
            self.session.commit()
            
        except SQLAlchemyError as e:
            self.session.rollback()
            raise DatabaseError(
                "Failed to add weather data",
                {'original_error': str(e)}
            )

    @error_handler
    def add_energy_data(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> None:
        """Add energy consumption data to database with validation"""
        try:
            if isinstance(data, pd.DataFrame):
                validate_dataframe(
                    data,
                    required_columns=['timestamp', 'consumption'],
                    numeric_columns=['consumption'],
                    datetime_columns=['timestamp']
                )
                energy_records = data.to_dict('records')
            else:
                energy_records = [data]

            for record in energy_records:
                energy_entry = EnergyConsumption(**record)
                self.session.add(energy_entry)
            
            self.session.commit()
            
        except SQLAlchemyError as e:
            self.session.rollback()
            raise DatabaseError(
                "Failed to add energy consumption data",
                {'original_error': str(e)}
            )

    @error_handler
    def get_weather_data(self, 
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Retrieve weather data with date range filtering"""
        try:
            query = self.session.query(WeatherData)
            
            if start_date:
                query = query.filter(WeatherData.timestamp >= start_date)
            if end_date:
                query = query.filter(WeatherData.timestamp <= end_date)
                
            results = query.all()
            if not results:
                raise DatabaseError(
                    "No weather data found for specified date range",
                    {'start_date': start_date, 'end_date': end_date}
                )
                
            return pd.DataFrame([r.__dict__ for r in results])
            
        except SQLAlchemyError as e:
            raise DatabaseError(
                "Failed to retrieve weather data",
                {'original_error': str(e)}
            )

    @error_handler
    def get_energy_data(self,
                       start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Retrieve energy consumption data with date range filtering"""
        try:
            query = self.session.query(EnergyConsumption)
            
            if start_date:
                query = query.filter(EnergyConsumption.timestamp >= start_date)
            if end_date:
                query = query.filter(EnergyConsumption.timestamp <= end_date)
                
            results = query.all()
            if not results:
                raise DatabaseError(
                    "No energy consumption data found for specified date range",
                    {'start_date': start_date, 'end_date': end_date}
                )
                
            return pd.DataFrame([r.__dict__ for r in results])
            
        except SQLAlchemyError as e:
            raise DatabaseError(
                "Failed to retrieve energy consumption data",
                {'original_error': str(e)}
            )

    @error_handler
    def update_model_metrics(self, model_id: str, metrics: Dict[str, float]) -> None:
        """Update model metrics with validation"""
        try:
            existing_metrics = self.session.query(ModelMetrics).filter_by(
                model_id=model_id
            ).first()
            
            if not existing_metrics:
                metrics_entry = ModelMetrics(model_id=model_id, **metrics)
                self.session.add(metrics_entry)
            else:
                for key, value in metrics.items():
                    setattr(existing_metrics, key, value)
                    
            self.session.commit()
            
        except SQLAlchemyError as e:
            self.session.rollback()
            raise DatabaseError(
                "Failed to update model metrics",
                {'model_id': model_id, 'original_error': str(e)}
            )

    @error_handler
    def get_model_metrics(self, model_id: str) -> Dict[str, float]:
        """Retrieve model metrics"""
        try:
            metrics = self.session.query(ModelMetrics).filter_by(
                model_id=model_id
            ).first()
            
            if not metrics:
                raise DatabaseError(
                    "No metrics found for specified model",
                    {'model_id': model_id}
                )
                
            return {
                key: value for key, value in metrics.__dict__.items()
                if not key.startswith('_')
            }
            
        except SQLAlchemyError as e:
            raise DatabaseError(
                "Failed to retrieve model metrics",
                {'model_id': model_id, 'original_error': str(e)}
            )

    @error_handler
    def delete_old_data(self, days_to_keep: int) -> None:
        """Delete data older than specified days with safety checks"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # Count records to be deleted
            weather_count = self.session.query(WeatherData).filter(
                WeatherData.timestamp < cutoff_date
            ).count()
            
            energy_count = self.session.query(EnergyConsumption).filter(
                EnergyConsumption.timestamp < cutoff_date
            ).count()
            
            # Log deletion counts
            self.logger.info(f"Deleting {weather_count} weather records and "
                           f"{energy_count} energy records")
            
            # Perform deletions
            self.session.query(WeatherData).filter(
                WeatherData.timestamp < cutoff_date
            ).delete()
            
            self.session.query(EnergyConsumption).filter(
                EnergyConsumption.timestamp < cutoff_date
            ).delete()
            
            self.session.commit()
            
        except SQLAlchemyError as e:
            self.session.rollback()
            raise DatabaseError(
                "Failed to delete old data",
                {
                    'days_to_keep': days_to_keep,
                    'cutoff_date': cutoff_date,
                    'original_error': str(e)
                }
            )

    @error_handler
    def backup_database(self, backup_path: str) -> None:
        """Create database backup"""
        try:
            # Export tables to CSV
            tables = {
                'weather_data': WeatherData,
                'energy_consumption': EnergyConsumption,
                'model_metrics': ModelMetrics
            }
            
            for name, table in tables.items():
                data = pd.read_sql(
                    self.session.query(table).statement,
                    self.session.bind
                )
                data.to_csv(f"{backup_path}/{name}.csv", index=False)
                
        except Exception as e:
            raise DatabaseError(
                "Failed to create database backup",
                {'backup_path': backup_path, 'original_error': str(e)}
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.session.rollback()
        self.session.close()
