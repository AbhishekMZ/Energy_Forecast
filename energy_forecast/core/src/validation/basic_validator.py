import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BasicDataValidator:
    """Basic validation checks for energy forecast data"""
    
    def __init__(self):
        # Define acceptable ranges for different parameters
        self.valid_ranges = {
            'temperature': (-10, 50),        # Celsius
            'humidity': (0, 100),            # Percentage
            'precipitation': (0, 200),       # mm (increased for extreme events)
            'cloud_cover': (0, 100),         # Percentage
            'wind_speed': (0, 100),          # km/h (increased for gusts)
            'wind_direction': (0, 360),      # degrees
            'solar_radiation': (0, 1200),    # W/m² (increased for peak hours)
            'total_demand': (0, 5000),       # MW (increased for peaks)
            'total_supply': (0, 6000),       # MW (increased for peaks)
            'distribution_loss': (0, 0.3)    # 0-30%
        }
        
        # Define tolerances for value range checks
        self.range_tolerances = {
            'temperature': 2,       # 2°C tolerance
            'humidity': 5,          # 5% tolerance
            'precipitation': 10,    # 10mm tolerance
            'cloud_cover': 5,       # 5% tolerance
            'wind_speed': 5,        # 5km/h tolerance
            'solar_radiation': 50,  # 50W/m² tolerance
            'total_demand': 100,    # 100MW tolerance
            'total_supply': 100,    # 100MW tolerance
            'distribution_loss': 0.02  # 2% tolerance
        }
        
        self.required_columns = [
            'timestamp', 'city', 'temperature', 'humidity', 'precipitation',
            'cloud_cover', 'wind_speed', 'wind_direction', 'solar_radiation',
            'total_demand', 'total_supply', 'distribution_loss'
        ]
        
        self.valid_cities = [
            'Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata',
            'Hyderabad', 'Pune'
        ]
    
    def validate_data_structure(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate basic data structure and required columns"""
        errors = []
        
        # Check if DataFrame is empty
        if df.empty:
            errors.append("DataFrame is empty")
            return False, errors
        
        # Check required columns
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Check data types
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            errors.append("timestamp column is not datetime type")
        
        if not df['city'].dtype == 'object':
            errors.append("city column is not string type")
        
        # Check for invalid cities
        invalid_cities = set(df['city'].unique()) - set(self.valid_cities)
        if invalid_cities:
            errors.append(f"Invalid cities found: {invalid_cities}")
        
        return len(errors) == 0, errors
    
    def validate_value_ranges(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, List[str]]]:
        """Validate that values fall within acceptable ranges"""
        errors = {}
        
        for column, (min_val, max_val) in self.valid_ranges.items():
            if column in df.columns:
                # Check for NaN values
                nan_count = df[column].isna().sum()
                if nan_count > 0:
                    errors[column] = [f"Found {nan_count} NaN values"]
                
                # Get tolerance for this column
                tolerance = self.range_tolerances.get(column, 0)
                
                # Check for values outside valid range (with tolerance)
                invalid_mask = (df[column] < min_val - tolerance) | (df[column] > max_val + tolerance)
                invalid_count = invalid_mask.sum()
                
                if invalid_count > 0:
                    error_msg = f"Found {invalid_count} values outside range [{min_val}, {max_val}] (tolerance: ±{tolerance})"
                    if column in errors:
                        errors[column].append(error_msg)
                    else:
                        errors[column] = [error_msg]
                        
                # Additional check for specific columns
                if column == 'total_supply':
                    # Supply should be greater than or equal to demand (with tolerance)
                    supply_tolerance = self.range_tolerances['total_supply']
                    invalid_supply = (df['total_supply'] < df['total_demand'] - supply_tolerance).sum()
                    if invalid_supply > 0:
                        error_msg = f"Found {invalid_supply} instances where supply is less than demand (tolerance: {supply_tolerance}MW)"
                        if column in errors:
                            errors[column].append(error_msg)
                        else:
                            errors[column] = [error_msg]
                
                # Check for sudden changes
                if column not in ['wind_direction', 'precipitation']:  # Exclude columns that can change rapidly
                    max_change = df[column].diff().abs().max()
                    if max_change > (max_val - min_val) * 0.5:  # If change is more than 50% of range
                        error_msg = f"Found sudden change of {max_change:.2f} units"
                        if column in errors:
                            errors[column].append(error_msg)
                        else:
                            errors[column] = [error_msg]
        
        return len(errors) == 0, errors
    
    def validate_temporal_consistency(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate temporal aspects of the data"""
        errors = []
        
        # Sort by timestamp
        df_sorted = df.sort_values('timestamp')
        
        # Check for duplicate timestamps per city
        duplicates = df.groupby('city')['timestamp'].apply(lambda x: x.duplicated().sum())
        if duplicates.sum() > 0:
            errors.append(f"Found duplicate timestamps: {duplicates[duplicates > 0].to_dict()}")
        
        # Check for gaps in time series
        for city in df['city'].unique():
            city_data = df_sorted[df_sorted['city'] == city]
            time_diff = city_data['timestamp'].diff()
            
            # Check for gaps larger than 1 hour
            large_gaps = time_diff[time_diff > timedelta(hours=1)]
            if not large_gaps.empty:
                errors.append(f"Found {len(large_gaps)} time gaps > 1 hour in {city}")
        
        return len(errors) == 0, errors
    
    def validate_consistency_rules(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate business logic and consistency rules"""
        errors = []
        
        # Check if total demand components sum up correctly
        demand_components = ['residential_demand', 'commercial_demand', 'industrial_demand']
        if all(col in df.columns for col in demand_components):
            total = df[demand_components].sum(axis=1)
            diff = abs(total - df['total_demand'])
            tolerance = self.range_tolerances['total_demand']
            inconsistent = (diff > tolerance).sum()
            if inconsistent > 0:
                errors.append(f"Found {inconsistent} rows where demand components don't sum to total (tolerance: {tolerance}MW)")
        
        # Check for realistic daily patterns
        for city in df['city'].unique():
            city_data = df[df['city'] == city]
            
            # Check if peak demand occurs during expected hours (7-10 or 18-22)
            peak_hours = list(range(7, 11)) + list(range(18, 23))
            peak_demand_hours = city_data.groupby(city_data['timestamp'].dt.hour)['total_demand'].mean()
            max_hour = peak_demand_hours.idxmax()
            
            if max_hour not in peak_hours:
                errors.append(f"{city}: Unexpected peak demand hour {max_hour}")
            
            # Check if solar radiation follows daylight pattern
            if 'solar_radiation' in df.columns:
                night_solar = city_data[
                    (city_data['timestamp'].dt.hour < 6) | 
                    (city_data['timestamp'].dt.hour > 19)
                ]['solar_radiation']
                
                # Allow for small values during twilight hours
                tolerance = self.range_tolerances['solar_radiation']
                night_violations = (night_solar > tolerance).sum()
                
                if night_violations > 0:
                    errors.append(f"{city}: Found {night_violations} instances of solar radiation > {tolerance}W/m² during night hours")
            
            # Check temperature-demand relationship
            if len(city_data) >= 24:  # At least a day's worth of data
                temp_demand_corr = city_data['temperature'].corr(city_data['total_demand'])
                if temp_demand_corr < 0:  # Negative correlation is suspicious
                    errors.append(f"{city}: Unexpected negative correlation between temperature and demand: {temp_demand_corr:.2f}")
        
        return len(errors) == 0, errors
    
    def run_all_validations(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, any]]:
        """Run all validation checks and return comprehensive results"""
        validation_results = {}
        
        # Structure validation
        structure_valid, structure_errors = self.validate_data_structure(df)
        validation_results['structure'] = {
            'valid': structure_valid,
            'errors': structure_errors
        }
        
        # Value range validation
        ranges_valid, range_errors = self.validate_value_ranges(df)
        validation_results['value_ranges'] = {
            'valid': ranges_valid,
            'errors': range_errors
        }
        
        # Temporal consistency validation
        temporal_valid, temporal_errors = self.validate_temporal_consistency(df)
        validation_results['temporal'] = {
            'valid': temporal_valid,
            'errors': temporal_errors
        }
        
        # Business logic validation
        consistency_valid, consistency_errors = self.validate_consistency_rules(df)
        validation_results['consistency'] = {
            'valid': consistency_valid,
            'errors': consistency_errors
        }
        
        # Overall validation result
        all_valid = all([
            structure_valid,
            ranges_valid,
            temporal_valid,
            consistency_valid
        ])
        
        return all_valid, validation_results

    def generate_validation_report(self, validation_results: Dict) -> str:
        """Generate a formatted validation report"""
        report = []
        report.append("=== Data Validation Report ===\n")
        
        # Overall status
        overall_valid = all(result['valid'] for result in validation_results.values())
        report.append(f"Overall Status: {'✓ PASSED' if overall_valid else '✗ FAILED'}\n")
        
        # Detailed results for each validation type
        for validation_type, results in validation_results.items():
            report.append(f"\n{validation_type.upper()} Validation:")
            report.append(f"Status: {'✓ PASSED' if results['valid'] else '✗ FAILED'}")
            
            if not results['valid']:
                report.append("Errors found:")
                if isinstance(results['errors'], dict):
                    for key, errors in results['errors'].items():
                        for error in errors:
                            report.append(f"  - {key}: {error}")
                else:
                    for error in results['errors']:
                        report.append(f"  - {error}")
            
            report.append("")
        
        return "\n".join(report)
