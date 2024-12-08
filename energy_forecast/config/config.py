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
    
    # Redis settings
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    REDIS_TTL = int(os.getenv("REDIS_TTL", 3600))  # 1 hour default

    # Batch processing settings
    MODEL_BATCH_SIZE = int(os.getenv("MODEL_BATCH_SIZE", 32))
    ENDPOINT_BATCH_SIZE = int(os.getenv("ENDPOINT_BATCH_SIZE", 50))
    MAX_BATCH_WAIT_TIME = float(os.getenv("MAX_BATCH_WAIT_TIME", 0.1))

    # Performance monitoring
    ENABLE_PERFORMANCE_MONITORING = os.getenv("ENABLE_PERFORMANCE_MONITORING", "true").lower() == "true"
    METRICS_COLLECTION_INTERVAL = int(os.getenv("METRICS_COLLECTION_INTERVAL", 60))  # seconds

    # Logging configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = 'app.log'
