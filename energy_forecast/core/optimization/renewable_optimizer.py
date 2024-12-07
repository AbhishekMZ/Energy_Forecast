"""Renewable energy optimization module"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import logging

from energy_forecast.config.constants import CITIES, RENEWABLE_SOURCES
from energy_forecast.core.utils.error_handling import OptimizationError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RenewableOptimizer:
    """Optimize renewable energy allocation based on availability and demand"""
    
    def __init__(self, city: str):
        self.city = city
        self.city_data = CITIES[city]
        self.renewable_sources = RENEWABLE_SOURCES
        
    def optimize_energy_mix(
        self,
        demand: float,
        weather_data: pd.Series,
        current_capacity: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Optimize energy mix prioritizing renewables
        
        Args:
            demand: Predicted energy demand (MW)
            weather_data: Current weather conditions
            current_capacity: Available capacity for each source
        
        Returns:
            Dict with allocated energy from each source
        """
        try:
            # Create optimization model
            model = pyo.ConcreteModel()
            
            # Decision variables for each energy source
            model.energy_vars = pyo.Var(
                self.renewable_sources.keys(),
                domain=pyo.NonNegativeReals
            )
            
            # Objective: Maximize renewable usage while minimizing cost
            def objective_rule(model):
                return sum(
                    model.energy_vars[source] * self.renewable_sources[source]['priority_weight']
                    for source in self.renewable_sources
                )
            model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)
            
            # Constraint 1: Meet demand
            def demand_rule(model):
                return sum(model.energy_vars[source] for source in self.renewable_sources) == demand
            model.demand_constraint = pyo.Constraint(rule=demand_rule)
            
            # Constraint 2: Source capacity limits
            def capacity_rule(model, source):
                max_capacity = self._calculate_available_capacity(
                    source,
                    weather_data,
                    current_capacity[source]
                )
                return model.energy_vars[source] <= max_capacity
            model.capacity_constraints = pyo.Constraint(
                self.renewable_sources.keys(),
                rule=capacity_rule
            )
            
            # Solve optimization problem
            solver = SolverFactory('glpk')
            results = solver.solve(model)
            
            if results.solver.status == pyo.SolverStatus.ok:
                # Extract solution
                solution = {
                    source: pyo.value(model.energy_vars[source])
                    for source in self.renewable_sources
                }
                return solution
            else:
                raise OptimizationError("Failed to find optimal solution")
                
        except Exception as e:
            raise OptimizationError(
                "Error in renewable optimization",
                {'original_error': str(e)}
            )
    
    def _calculate_available_capacity(
        self,
        source: str,
        weather_data: pd.Series,
        base_capacity: float
    ) -> float:
        """Calculate available capacity based on weather conditions"""
        source_config = self.renewable_sources[source]
        
        # Base availability factor
        availability = source_config['base_availability']
        
        # Apply weather factors
        for weather_param, impact in source_config['weather_impact'].items():
            if weather_param in weather_data:
                param_value = weather_data[weather_param]
                availability *= self._calculate_weather_impact(
                    param_value,
                    impact['optimal_range'],
                    impact['impact_factor']
                )
        
        return base_capacity * availability
    
    def _calculate_weather_impact(
        self,
        value: float,
        optimal_range: Tuple[float, float],
        impact_factor: float
    ) -> float:
        """Calculate weather impact on energy source availability"""
        min_val, max_val = optimal_range
        
        if min_val <= value <= max_val:
            return 1.0
        elif value < min_val:
            return 1.0 - (min_val - value) * impact_factor
        else:
            return 1.0 - (value - max_val) * impact_factor
    
    def get_production_schedule(
        self,
        energy_mix: Dict[str, float],
        start_time: datetime
    ) -> pd.DataFrame:
        """Generate production schedule based on optimized energy mix"""
        try:
            schedule = []
            current_time = start_time
            
            for source, amount in energy_mix.items():
                if amount > 0:
                    source_config = self.renewable_sources[source]
                    ramp_rate = source_config['ramp_rate']  # MW per hour
                    startup_time = source_config['startup_time']  # hours
                    
                    # Calculate production timeline
                    schedule.append({
                        'source': source,
                        'amount': amount,
                        'start_time': current_time + timedelta(hours=startup_time),
                        'ramp_up_time': amount / ramp_rate,
                        'cost_per_mwh': source_config['cost_per_mwh']
                    })
            
            return pd.DataFrame(schedule)
            
        except Exception as e:
            raise OptimizationError(
                "Error generating production schedule",
                {'original_error': str(e)}
            )
