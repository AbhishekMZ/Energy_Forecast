# Configuration Guide

## Overview

This guide details all configuration options and settings for the Energy Forecast Platform, including environment variables, configuration files, and runtime settings.

## Environment Configuration

### 1. Environment Variables

```bash
# .env.example
# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/energy_forecast
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=10

# Redis Configuration
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=your_redis_password
REDIS_SSL=false

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_TIMEOUT=30
API_DEBUG=false

# Security
SECRET_KEY=your_secret_key
JWT_ALGORITHM=HS256
JWT_EXPIRY=3600
CORS_ORIGINS=["http://localhost:3000"]

# Model Configuration
MODEL_PATH=/path/to/models
MODEL_CACHE_SIZE=1000
MODEL_BATCH_SIZE=64

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
LOG_LEVEL=INFO

# External Services
WEATHER_API_KEY=your_weather_api_key
WEATHER_API_URL=https://api.weather.com/v1
```

### 2. Application Configuration

```python
# config/settings.py
from pydantic import BaseSettings, PostgresDsn, RedisDsn

class Settings(BaseSettings):
    """Application settings."""
    
    # Database
    database_url: PostgresDsn
    database_pool_size: int = 20
    database_max_overflow: int = 10
    
    # Redis
    redis_url: RedisDsn
    redis_password: str
    redis_ssl: bool = False
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    api_timeout: int = 30
    api_debug: bool = False
    
    # Security
    secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_expiry: int = 3600
    cors_origins: list[str]
    
    # Model
    model_path: str
    model_cache_size: int = 1000
    model_batch_size: int = 64
    
    # Monitoring
    prometheus_port: int = 9090
    grafana_port: int = 3000
    log_level: str = "INFO"
    
    # External Services
    weather_api_key: str
    weather_api_url: str
    
    class Config:
        env_file = ".env"
```

## Security Configuration

### 1. Authentication Settings

```python
# config/auth.py
from datetime import timedelta

AUTH_CONFIG = {
    'token_expiry': timedelta(hours=1),
    'refresh_token_expiry': timedelta(days=7),
    'password_hash_algorithm': 'bcrypt',
    'password_hash_rounds': 12,
    'session_cookie_name': 'session_id',
    'session_cookie_secure': True,
    'session_cookie_httponly': True,
    'session_cookie_samesite': 'Lax'
}

# Token validation settings
TOKEN_VALIDATION = {
    'verify_signature': True,
    'verify_exp': True,
    'verify_nbf': True,
    'verify_iat': True,
    'verify_aud': True,
    'require_exp': True,
    'require_iat': True,
    'require_nbf': False
}
```

### 2. Security Headers

```python
# config/security.py
SECURITY_HEADERS = {
    'X-Frame-Options': 'DENY',
    'X-Content-Type-Options': 'nosniff',
    'X-XSS-Protection': '1; mode=block',
    'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
    'Content-Security-Policy': "default-src 'self'",
    'Referrer-Policy': 'strict-origin-when-cross-origin'
}

# CORS configuration
CORS_CONFIG = {
    'allow_origins': ['http://localhost:3000'],
    'allow_methods': ['GET', 'POST', 'PUT', 'DELETE'],
    'allow_headers': ['*'],
    'allow_credentials': True,
    'max_age': 3600
}
```

## Database Configuration

### 1. Connection Settings

```python
# config/database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_CONFIG = {
    'pool_size': 20,
    'max_overflow': 10,
    'pool_timeout': 30,
    'pool_recycle': 1800,
    'echo': False,
    'echo_pool': False,
    'isolation_level': 'READ COMMITTED'
}

def create_database_engine(url: str):
    """Create database engine with configuration."""
    return create_engine(
        url,
        **DATABASE_CONFIG,
        json_serializer=custom_json_serializer,
        json_deserializer=custom_json_deserializer
    )
```

### 2. Migration Settings

```python
# alembic.ini
[alembic]
script_location = migrations
sqlalchemy.url = driver://user:pass@localhost/dbname

# Logging configuration
[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic
```

## Caching Configuration

### 1. Redis Settings

```python
# config/cache.py
REDIS_CONFIG = {
    'encoding': 'utf-8',
    'decode_responses': True,
    'socket_timeout': 5,
    'socket_connect_timeout': 5,
    'retry_on_timeout': True,
    'health_check_interval': 30
}

# Cache TTL settings
CACHE_TTL = {
    'forecast': 300,  # 5 minutes
    'historical_data': 3600,  # 1 hour
    'model_metadata': 86400,  # 1 day
    'city_data': 86400,  # 1 day
    'weather_data': 1800  # 30 minutes
}
```

### 2. Cache Policies

```python
# config/cache_policy.py
CACHE_POLICIES = {
    'forecast': {
        'strategy': 'write-through',
        'ttl': 300,
        'max_size': 1000,
        'eviction_policy': 'lru'
    },
    'historical': {
        'strategy': 'write-back',
        'ttl': 3600,
        'max_size': 5000,
        'eviction_policy': 'lfu'
    }
}
```

## Model Configuration

### 1. Model Settings

```python
# config/model.py
MODEL_CONFIG = {
    'lstm': {
        'layers': 2,
        'units': 64,
        'dropout': 0.2,
        'recurrent_dropout': 0.2,
        'optimizer': 'adam',
        'loss': 'mse',
        'metrics': ['mae', 'mape']
    },
    'xgboost': {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'objective': 'reg:squarederror',
        'booster': 'gbtree',
        'subsample': 0.8
    },
    'transformer': {
        'num_layers': 4,
        'num_heads': 8,
        'd_model': 256,
        'dff': 1024,
        'dropout_rate': 0.1,
        'optimizer': 'adam'
    }
}
```

### 2. Training Settings

```python
# config/training.py
TRAINING_CONFIG = {
    'batch_size': 64,
    'epochs': 100,
    'validation_split': 0.2,
    'early_stopping': {
        'monitor': 'val_loss',
        'patience': 10,
        'min_delta': 0.001
    },
    'model_checkpoint': {
        'monitor': 'val_loss',
        'save_best_only': True,
        'mode': 'min'
    }
}
```

## Monitoring Configuration

### 1. Logging Settings

```python
# config/logging.py
import logging.config

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
        'json': {
            'class': 'pythonjsonlogger.jsonlogger.JsonFormatter',
            'format': '%(asctime)s %(levelname)s %(name)s %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'level': 'INFO'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'logs/app.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'formatter': 'json',
            'level': 'INFO'
        }
    },
    'loggers': {
        '': {
            'handlers': ['console', 'file'],
            'level': 'INFO'
        }
    }
}
```

### 2. Metrics Settings

```python
# config/metrics.py
METRICS_CONFIG = {
    'enabled': True,
    'host': '0.0.0.0',
    'port': 9090,
    'path': '/metrics',
    'collectors': {
        'process': True,
        'platform': True,
        'gc': True
    },
    'labels': {
        'application': 'energy_forecast',
        'environment': 'production'
    }
}

# Prometheus configuration
PROMETHEUS_CONFIG = {
    'scrape_interval': '15s',
    'evaluation_interval': '15s',
    'scrape_timeout': '10s',
    'static_configs': [{
        'targets': ['localhost:9090']
    }]
}
```

## API Configuration

### 1. FastAPI Settings

```python
# config/api.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

API_CONFIG = {
    'title': 'Energy Forecast API',
    'description': 'API for energy consumption forecasting',
    'version': '1.0.0',
    'docs_url': '/docs',
    'redoc_url': '/redoc',
    'openapi_url': '/openapi.json'
}

# API rate limiting
RATE_LIMIT_CONFIG = {
    'default': {
        'times': 100,
        'seconds': 60
    },
    'forecast': {
        'times': 50,
        'seconds': 60
    },
    'historical': {
        'times': 30,
        'seconds': 60
    }
}

# API timeout settings
TIMEOUT_CONFIG = {
    'default': 30,
    'forecast': 60,
    'historical': 120,
    'training': 300
}
```

### 2. Middleware Settings

```python
# config/middleware.py
MIDDLEWARE_CONFIG = {
    'compression': {
        'minimum_size': 500,
        'compression_level': 6
    },
    'cors': {
        'allow_origins': ['*'],
        'allow_methods': ['*'],
        'allow_headers': ['*']
    },
    'security': {
        'ssl_redirect': True,
        'force_ssl': True,
        'frame_deny': True
    }
}
```

## Deployment Configuration

### 1. Docker Settings

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/energy_forecast
      - REDIS_URL=redis://redis:6379/0
    ports:
      - "8000:8000"
    depends_on:
      - db
      - redis

  db:
    image: postgres:13
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=energy_forecast
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:6
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

### 2. Kubernetes Settings

```yaml
# kubernetes/config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: energy-forecast-config
data:
  DATABASE_URL: postgresql://user:password@postgres:5432/energy_forecast
  REDIS_URL: redis://redis:6379/0
  API_HOST: 0.0.0.0
  API_PORT: "8000"
  LOG_LEVEL: INFO

---
apiVersion: v1
kind: Secret
metadata:
  name: energy-forecast-secrets
type: Opaque
data:
  DATABASE_PASSWORD: base64_encoded_password
  REDIS_PASSWORD: base64_encoded_password
  SECRET_KEY: base64_encoded_key
```

## Additional Resources

- [API Documentation](./api_reference.md)
- [Deployment Guide](./deployment_guide.md)
- [Security Guide](./security_guide.md)
- [Development Setup](./development_setup.md)
