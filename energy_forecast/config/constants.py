"""Constants and configuration values for the energy forecast system"""

# Validation thresholds
VALIDATION_THRESHOLDS = {
    'temperature': {'min': -10, 'max': 50, 'tolerance': 2},
    'humidity': {'min': 0, 'max': 100, 'tolerance': 5},
    'precipitation': {'min': 0, 'max': 200, 'tolerance': 10},
    'cloud_cover': {'min': 0, 'max': 100, 'tolerance': 5},
    'wind_speed': {'min': 0, 'max': 100, 'tolerance': 5},
    'solar_radiation': {'min': 0, 'max': 1200, 'tolerance': 50},
    'total_demand': {'min': 0, 'max': 5000, 'tolerance': 100},
    'total_supply': {'min': 0, 'max': 6000, 'tolerance': 100},
    'distribution_loss': {'min': 0, 'max': 30, 'tolerance': 2}
}

# Statistical validation thresholds
STATISTICAL_THRESHOLDS = {
    'correlation': {
        'temperature_demand': {'min': 0.3, 'max': 0.95},
        'solar_demand': {'min': 0.3, 'max': 0.95},
        'cloud_solar': {'min': -0.95, 'max': -0.3},
        'temperature_humidity': {'min': -0.95, 'max': -0.3}
    },
    'outliers': {
        'zscore_threshold': 3,
        'iqr_multiplier': 1.5
    },
    'stationarity': {
        'mean_change_threshold': 1.5,
        'std_change_threshold': 1.5
    }
}

# Model configuration thresholds
MODEL_THRESHOLDS = {
    'min_training_samples': 1000,
    'max_missing_ratio': 0.1,
    'min_r2_score': 0.7,
    'max_rmse': 500,
    'cross_validation_folds': 5
}

# Data pipeline configuration
PIPELINE_CONFIG = {
    'fill_methods': {
        'temperature': 'interpolate',
        'humidity': 'interpolate',
        'precipitation': 'mean',
        'cloud_cover': 'interpolate',
        'wind_speed': 'interpolate',
        'solar_radiation': 'interpolate',
        'total_demand': 'interpolate',
        'total_supply': 'interpolate',
        'distribution_loss': 'mean'
    },
    'scaling': {
        'temperature': 'standard',
        'humidity': 'minmax',
        'precipitation': 'minmax',
        'cloud_cover': 'minmax',
        'wind_speed': 'standard',
        'solar_radiation': 'standard',
        'total_demand': 'standard',
        'total_supply': 'standard'
    }
}

# Cities and their metadata
CITIES = {
    'Mumbai': {
        'latitude': 19.0760,
        'longitude': 72.8777,
        'timezone': 'Asia/Kolkata',
        'population': 20411274,
        'energy_sources': {
            'solar': 2000,  # MW capacity
            'wind': 1500,
            'hydro': 1000,
            'biomass': 500,
            'natural_gas': 3000
        },
        'peak_demand_factor': 1.5,
        'base_demand': 2500  # MW
    },
    'Delhi': {
        'latitude': 28.6139,
        'longitude': 77.2090,
        'timezone': 'Asia/Kolkata',
        'population': 30290936,
        'energy_sources': {
            'solar': 2500,
            'wind': 1000,
            'hydro': 1500,
            'biomass': 800,
            'natural_gas': 4000
        },
        'peak_demand_factor': 1.6,
        'base_demand': 3500
    }
}

# Renewable Energy Sources Configuration
RENEWABLE_SOURCES = {
    'solar': {
        'priority_weight': 1.0,  # Highest priority
        'base_availability': 0.95,
        'weather_impact': {
            'cloud_cover': {
                'optimal_range': (0, 20),
                'impact_factor': 0.01
            },
            'solar_radiation': {
                'optimal_range': (600, 1000),
                'impact_factor': 0.002
            }
        },
        'ramp_rate': 100,  # MW per hour
        'startup_time': 0.5,  # hours
        'cost_per_mwh': 40  # USD
    },
    'wind': {
        'priority_weight': 0.9,
        'base_availability': 0.90,
        'weather_impact': {
            'wind_speed': {
                'optimal_range': (10, 40),
                'impact_factor': 0.05
            }
        },
        'ramp_rate': 50,
        'startup_time': 0.25,
        'cost_per_mwh': 45
    },
    'hydro': {
        'priority_weight': 0.8,
        'base_availability': 0.98,
        'weather_impact': {
            'precipitation': {
                'optimal_range': (0, 50),
                'impact_factor': 0.01
            }
        },
        'ramp_rate': 200,
        'startup_time': 1,
        'cost_per_mwh': 35
    },
    'biomass': {
        'priority_weight': 0.7,
        'base_availability': 0.85,
        'weather_impact': {},  # Not significantly affected by weather
        'ramp_rate': 30,
        'startup_time': 2,
        'cost_per_mwh': 55
    },
    'natural_gas': {  # Backup source
        'priority_weight': 0.3,
        'base_availability': 0.98,
        'weather_impact': {},
        'ramp_rate': 150,
        'startup_time': 1,
        'cost_per_mwh': 70
    }
}
