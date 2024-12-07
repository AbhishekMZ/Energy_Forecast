import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Update imports to use absolute imports from energy_forecast package
from energy_forecast.core.validation.basic_validator import BasicDataValidator
from energy_forecast.core.validation.statistical_validator import StatisticalValidator

@pytest.fixture
def sample_valid_data():
    """Create a sample valid dataset"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-01-07', freq='h')
    cities = ['Mumbai', 'Delhi']
    
    data = []
    for city in cities:
        for date in dates:
            hour = date.hour
            # Generate realistic patterns
            temp_base = 25 + 5 * np.sin(np.pi * hour / 12)  # Daily temperature pattern
            demand_base = 2000 + 800 * np.sin(np.pi * hour / 12)  # Daily demand pattern
            solar_base = max(0, 800 * np.sin(np.pi * hour / 12))  # Solar radiation pattern
            
            # Add seasonal component
            day_of_year = date.dayofyear
            seasonal_temp = 5 * np.sin(2 * np.pi * day_of_year / 365)
            seasonal_demand = 500 * np.sin(2 * np.pi * day_of_year / 365)
            
            # Add weather impact on demand
            temp = temp_base + seasonal_temp + np.random.normal(0, 0.5)  # Reduced temperature noise
            weather_impact = 100 * (temp - 25)  # Higher demand for extreme temperatures
            demand = demand_base + seasonal_demand + weather_impact + np.random.normal(0, 20)  # Reduced demand noise
            
            # Ensure cloud cover and solar radiation are inversely correlated
            cloud_cover = 50 + 30 * np.sin(np.pi * hour / 12) + np.random.normal(0, 5)
            cloud_cover = max(0, min(100, cloud_cover))
            solar_radiation = max(0, solar_base * (1 - cloud_cover/100) + np.random.normal(0, 10))
            
            # Ensure humidity is inversely correlated with temperature
            humidity = 90 - temp + np.random.normal(0, 2)  # Higher humidity when cooler
            humidity = max(0, min(100, humidity))
            
            row = {
                'timestamp': date,
                'city': city,
                'temperature': temp,
                'humidity': humidity,
                'precipitation': max(0, np.random.exponential(1) if np.random.random() < 0.2 else 0),
                'cloud_cover': cloud_cover,
                'wind_speed': max(0, 15 + 5 * np.sin(np.pi * hour / 12) + np.random.normal(0, 1)),
                'wind_direction': np.random.uniform(0, 360),
                'solar_radiation': solar_radiation,
                'total_demand': max(0, demand),
                'total_supply': max(0, demand * 1.1 + np.random.normal(0, 10)),  # Supply > Demand
                'distribution_loss': np.random.uniform(0.05, 0.10)  # Tighter range
            }
            data.append(row)
    
    return pd.DataFrame(data)

@pytest.fixture
def sample_invalid_data(sample_valid_data):
    """Create a sample invalid dataset"""
    df = sample_valid_data.copy()
    
    # Introduce various errors
    df.loc[0, 'temperature'] = 100  # Invalid temperature
    df.loc[1, 'humidity'] = 150     # Invalid humidity
    df.loc[2, 'city'] = 'Invalid City'  # Invalid city
    df.loc[3, 'total_supply'] = 0   # Invalid supply
    df.loc[4:8, 'timestamp'] = pd.NaT  # Invalid timestamps
    
    return df

class TestBasicValidator:
    def test_valid_data_structure(self, sample_valid_data):
        validator = BasicDataValidator()
        valid, errors = validator.validate_data_structure(sample_valid_data)
        assert valid
        assert not errors
    
    def test_invalid_data_structure(self, sample_invalid_data):
        validator = BasicDataValidator()
        valid, errors = validator.validate_data_structure(sample_invalid_data)
        assert not valid
        assert any("Invalid cities found" in error for error in errors)
    
    def test_value_ranges(self, sample_valid_data):
        validator = BasicDataValidator()
        valid, errors = validator.validate_value_ranges(sample_valid_data)
        assert valid
        assert not errors
    
    def test_invalid_value_ranges(self, sample_invalid_data):
        validator = BasicDataValidator()
        valid, errors = validator.validate_value_ranges(sample_invalid_data)
        assert not valid
        assert 'temperature' in errors
        assert 'humidity' in errors
    
    def test_temporal_consistency(self, sample_valid_data):
        validator = BasicDataValidator()
        valid, errors = validator.validate_temporal_consistency(sample_valid_data)
        assert valid
        assert not errors
    
    def test_invalid_temporal_consistency(self, sample_invalid_data):
        validator = BasicDataValidator()
        valid, errors = validator.validate_temporal_consistency(sample_invalid_data)
        assert not valid
    
    def test_run_all_validations(self, sample_valid_data):
        validator = BasicDataValidator()
        valid, results = validator.run_all_validations(sample_valid_data)
        assert valid
        assert all(result['valid'] for result in results.values())
    
    def test_sudden_changes(self, sample_valid_data):
        """Test detection of sudden changes in values"""
        validator = BasicDataValidator()
        df = sample_valid_data.copy()
        
        # Introduce a sudden change in temperature
        df.loc[10, 'temperature'] = df.loc[9, 'temperature'] + 30
        valid, errors = validator.validate_value_ranges(df)
        assert not valid
        assert 'temperature' in errors
        assert any('sudden change' in error for error in errors['temperature'])
    
    def test_night_solar_radiation(self, sample_valid_data):
        """Test detection of solar radiation during night hours"""
        validator = BasicDataValidator()
        df = sample_valid_data.copy()
        
        # Add solar radiation during night hours
        night_mask = (df['timestamp'].dt.hour < 6) | (df['timestamp'].dt.hour > 19)
        df.loc[night_mask, 'solar_radiation'] = 500
        
        valid, errors = validator.validate_consistency_rules(df)
        assert not valid
        assert any('solar radiation' in error.lower() for error in errors)
    
    def test_temperature_demand_correlation(self, sample_valid_data):
        """Test validation of temperature-demand relationship"""
        validator = BasicDataValidator()
        df = sample_valid_data.copy()
        
        # Reverse the temperature-demand relationship
        df['total_demand'] = -df['total_demand']
        
        valid, errors = validator.validate_consistency_rules(df)
        assert not valid
        assert any('correlation' in error.lower() for error in errors)

class TestStatisticalValidator:
    def test_outlier_detection(self, sample_valid_data):
        validator = StatisticalValidator()
        outliers = validator.detect_outliers(sample_valid_data, 
                                          ['temperature', 'total_demand'])
        assert all(mask.sum() < len(sample_valid_data) * 0.1 
                  for mask in outliers.values())
    
    def test_statistical_patterns(self, sample_valid_data):
        validator = StatisticalValidator()
        valid, errors = validator.validate_statistical_patterns(sample_valid_data)
        assert valid
        assert not errors
    
    def test_distribution(self, sample_valid_data):
        validator = StatisticalValidator()
        valid, errors = validator.validate_distribution(sample_valid_data)
        assert valid
        assert not errors
    
    def test_run_statistical_validations(self, sample_valid_data):
        validator = StatisticalValidator()
        valid, results = validator.run_statistical_validations(sample_valid_data)
        assert valid
        assert all(result['valid'] for result in results.values())

@pytest.fixture
def seasonal_data():
    """Create sample data with seasonal patterns"""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='h')
    data = []
    
    for date in dates:
        hour = date.hour
        day_of_year = date.dayofyear
        
        # Base patterns
        daily_temp = 5 * np.sin(2 * np.pi * (hour / 24))
        daily_demand = 800 * np.sin(2 * np.pi * (hour / 24))
        
        # Seasonal patterns
        seasonal_temp = 25 + 10 * np.sin(2 * np.pi * (day_of_year / 365))
        seasonal_demand = 2000 + 500 * np.sin(2 * np.pi * (day_of_year / 365))
        
        # Combine patterns with noise
        temp = seasonal_temp + daily_temp + np.random.normal(0, 1)
        demand = seasonal_demand + daily_demand + np.random.normal(0, 50)
        
        # Add weather impact on demand
        weather_impact = 100 * (temp - 25)  # Higher demand for extreme temperatures
        demand += weather_impact
        
        row = {
            'timestamp': date,
            'city': 'Mumbai',
            'temperature': temp,
            'total_demand': max(0, demand),
            'solar_radiation': max(0, 800 * np.sin(np.pi * hour / 12) + np.random.normal(0, 20)),
            'total_supply': max(0, demand * 1.1 + np.random.normal(0, 20))
        }
        data.append(row)
    
    return pd.DataFrame(data)

def test_seasonal_patterns(seasonal_data):
    validator = StatisticalValidator()
    valid, errors = validator.validate_statistical_patterns(seasonal_data)
    assert valid
    assert not errors

def test_report_generation(sample_valid_data):
    basic_validator = BasicDataValidator()
    stat_validator = StatisticalValidator()
    
    # Test basic validation report
    _, basic_results = basic_validator.run_all_validations(sample_valid_data)
    basic_report = basic_validator.generate_validation_report(basic_results)
    assert isinstance(basic_report, str)
    assert "PASSED" in basic_report
    
    # Test statistical validation report
    _, stat_results = stat_validator.run_statistical_validations(sample_valid_data)
    stat_report = stat_validator.generate_statistical_report(stat_results)
    assert isinstance(stat_report, str)
    assert "PASSED" in stat_report
