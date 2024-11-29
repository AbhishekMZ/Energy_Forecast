import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Database configuration
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite:///energy_forecast.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # API Keys
    WEATHER_API_KEY = os.getenv('WEATHER_API_KEY', '')
    
    # Application configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    
    # Model parameters
    MODEL_PATH = os.path.join('models', 'saved')
    BATCH_SIZE = 32
    EPOCHS = 100
    
    # Data paths
    RAW_DATA_PATH = os.path.join('data', 'raw')
    PROCESSED_DATA_PATH = os.path.join('data', 'processed')
    
    # Logging configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = 'app.log'
