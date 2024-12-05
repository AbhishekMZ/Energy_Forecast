# Constants for Indian context
INDIAN_SEASONS = {
    "Winter": [11, 12, 1, 2],     # Nov to Feb
    "Summer": [3, 4, 5],          # Mar to May
    "Monsoon": [6, 7, 8, 9],      # Jun to Sep
    "Post-Monsoon": [10]          # October
}

INDIAN_HOLIDAYS = {
    "Republic_Day": "01-26",
    "Independence_Day": "08-15",
    "Gandhi_Jayanti": "10-02",
    # Add more holidays as needed
}

PEAK_HOURS = {
    "Morning": range(7, 11),      # 7 AM to 10 AM
    "Evening": range(18, 23)      # 6 PM to 10 PM
}

# City-specific baseline data
CITY_BASELINES = {
    "Mumbai": {
        "base_load": 3400,
        "peak_multiplier": 1.5,
        "industrial_ratio": 0.45,
        "commercial_ratio": 0.35,
        "residential_ratio": 0.20
    },
    "Delhi": {
        "base_load": 4000,
        "peak_multiplier": 1.6,
        "industrial_ratio": 0.35,
        "commercial_ratio": 0.40,
        "residential_ratio": 0.25
    },
    "Bangalore": {
        "base_load": 2500,
        "peak_multiplier": 1.4,
        "industrial_ratio": 0.40,
        "commercial_ratio": 0.45,
        "residential_ratio": 0.15
    },
    "Chennai": {
        "base_load": 2200,
        "peak_multiplier": 1.5,
        "industrial_ratio": 0.40,
        "commercial_ratio": 0.35,
        "residential_ratio": 0.25
    },
    "Kolkata": {
        "base_load": 2100,
        "peak_multiplier": 1.4,
        "industrial_ratio": 0.35,
        "commercial_ratio": 0.35,
        "residential_ratio": 0.30
    },
    "Hyderabad": {
        "base_load": 2300,
        "peak_multiplier": 1.45,
        "industrial_ratio": 0.40,
        "commercial_ratio": 0.40,
        "residential_ratio": 0.20
    },
    "Pune": {
        "base_load": 1800,
        "peak_multiplier": 1.35,
        "industrial_ratio": 0.45,
        "commercial_ratio": 0.30,
        "residential_ratio": 0.25
    }
}

# Infrastructure configurations
INFRASTRUCTURE_CONFIG = {
    "Mumbai": {
        "grid_capacity": 4000,  # MW
        "renewable_share": 0.10,
        "distribution_loss": 0.15,
        "backup_capacity": 1000,
        "smart_grid_coverage": 0.60
    },
    "Delhi": {
        "grid_capacity": 5500,
        "renewable_share": 0.15,
        "distribution_loss": 0.18,
        "backup_capacity": 1500,
        "smart_grid_coverage": 0.70
    },
    "Bangalore": {
        "grid_capacity": 3000,
        "renewable_share": 0.25,
        "distribution_loss": 0.12,
        "backup_capacity": 800,
        "smart_grid_coverage": 0.80
    },
    "Chennai": {
        "grid_capacity": 3500,
        "renewable_share": 0.20,
        "distribution_loss": 0.14,
        "backup_capacity": 900,
        "smart_grid_coverage": 0.65
    },
    "Kolkata": {
        "grid_capacity": 2800,
        "renewable_share": 0.12,
        "distribution_loss": 0.16,
        "backup_capacity": 700,
        "smart_grid_coverage": 0.55
    }
}

# Weather patterns for each city
WEATHER_PATTERNS = {
    "Mumbai": {
        "temp_range": (22, 35),
        "humidity_range": (60, 90),
        "rainfall_months": [6, 7, 8, 9],  # June to September
        "rainfall_intensity": "high",
        "seasonal_pattern": "coastal"
    },
    "Delhi": {
        "temp_range": (8, 45),
        "humidity_range": (30, 80),
        "rainfall_months": [7, 8, 9],
        "rainfall_intensity": "moderate",
        "seasonal_pattern": "continental"
    },
    "Bangalore": {
        "temp_range": (15, 35),
        "humidity_range": (40, 85),
        "rainfall_months": [5, 6, 9, 10],
        "rainfall_intensity": "moderate",
        "seasonal_pattern": "plateau"
    },
    "Chennai": {
        "temp_range": (24, 38),
        "humidity_range": (65, 85),
        "rainfall_months": [10, 11, 12],
        "rainfall_intensity": "very_high",
        "seasonal_pattern": "coastal"
    },
    "Kolkata": {
        "temp_range": (15, 40),
        "humidity_range": (50, 85),
        "rainfall_months": [6, 7, 8, 9],
        "rainfall_intensity": "high",
        "seasonal_pattern": "subtropical"
    }
}
