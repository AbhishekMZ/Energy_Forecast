"""Enhanced A/B Testing System with Statistical Analysis."""

import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple
import pandas as pd
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

@dataclass
class ExperimentConfig:
    name: str
    variants: Dict[str, float]
    metrics: List[str]
    min_sample_size: int
    confidence_level: float
    max_duration_days: int

@dataclass
class VariantResult:
    mean: float
    std: float
    sample_size: int
    confidence_interval: Tuple[float, float]
    p_value: float

class EnhancedABTesting:
    def __init__(self, db_connection):
        self.db = db_connection
        self.active_experiments = {}
        self.results_cache = {}
        
    def create_experiment(
        self,
        config: ExperimentConfig,
        metadata: Dict
    ) -> str:
        """Create new experiment with statistical parameters."""
        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.active_experiments[experiment_id] = {
            "config": config,
            "metadata": metadata,
            "start_time": datetime.now(),
            "status": "active",
            "results": {variant: [] for variant in config.variants.keys()},
            "statistical_significance": False
        }
        
        # Log experiment creation to database
        self._log_experiment_event(
            experiment_id,
            "creation",
            {"config": config.__dict__, "metadata": metadata}
        )
        
        return experiment_id
    
    def assign_variant(self, experiment_id: str, user_id: str) -> str:
        """Assign user to variant using consistent hashing."""
        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
            
        exp = self.active_experiments[experiment_id]
        variants = list(exp["config"].variants.keys())
        weights = list(exp["config"].variants.values())
        
        # Use hash of user_id for consistent assignment
        hash_value = hash(f"{experiment_id}:{user_id}")
        normalized_hash = (hash_value % 1000) / 1000.0
        
        cumulative_weights = np.cumsum(weights)
        for variant, threshold in zip(variants, cumulative_weights):
            if normalized_hash <= threshold:
                return variant
                
        return variants[-1]
    
    def log_variant_result(
        self,
        experiment_id: str,
        variant: str,
        metrics: Dict[str, float],
        user_id: Optional[str] = None
    ):
        """Log results with additional metadata."""
        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
            
        exp = self.active_experiments[experiment_id]
        if variant not in exp["config"].variants:
            raise ValueError(f"Variant {variant} not found")
            
        # Add result with metadata
        result = {
            "metrics": metrics,
            "timestamp": datetime.now(),
            "user_id": user_id
        }
        
        exp["results"][variant].append(result)
        self._log_experiment_event(
            experiment_id,
            "result",
            {"variant": variant, "metrics": metrics}
        )
        
        # Check if statistical significance achieved
        self._update_statistical_significance(experiment_id)
        
    def get_experiment_results(
        self,
        experiment_id: str,
        metric: Optional[str] = None
    ) -> Dict:
        """Get detailed experiment results with statistical analysis."""
        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
            
        exp = self.active_experiments[experiment_id]
        metrics_to_analyze = [metric] if metric else exp["config"].metrics
        
        results = {}
        for metric_name in metrics_to_analyze:
            variant_results = {}
            control_data = None
            
            for variant, data in exp["results"].items():
                if not data:
                    continue
                    
                values = [d["metrics"].get(metric_name) for d in data]
                values = [v for v in values if v is not None]
                
                if not values:
                    continue
                    
                stats_result = self._calculate_statistics(
                    values,
                    exp["config"].confidence_level
                )
                
                variant_results[variant] = stats_result
                if control_data is None:
                    control_data = values
                    
            results[metric_name] = {
                "variant_results": variant_results,
                "significant_difference": self._check_significance(
                    variant_results,
                    exp["config"].confidence_level
                )
            }
            
        return {
            "experiment_name": exp["config"].name,
            "duration": (datetime.now() - exp["start_time"]).total_seconds(),
            "sample_sizes": {
                variant: len(data)
                for variant, data in exp["results"].items()
            },
            "results": results,
            "visualization": self._generate_visualization(results)
        }
    
    def _calculate_statistics(
        self,
        values: List[float],
        confidence_level: float
    ) -> VariantResult:
        """Calculate detailed statistics for variant."""
        mean = np.mean(values)
        std = np.std(values)
        sample_size = len(values)
        
        # Calculate confidence interval
        ci = stats.t.interval(
            confidence_level,
            sample_size - 1,
            loc=mean,
            scale=std/np.sqrt(sample_size)
        )
        
        # Calculate p-value against normal distribution
        _, p_value = stats.normaltest(values)
        
        return VariantResult(
            mean=mean,
            std=std,
            sample_size=sample_size,
            confidence_interval=ci,
            p_value=p_value
        )
    
    def _check_significance(
        self,
        variant_results: Dict[str, VariantResult],
        confidence_level: float
    ) -> bool:
        """Check for statistical significance between variants."""
        if len(variant_results) < 2:
            return False
            
        variants = list(variant_results.keys())
        control = variants[0]
        treatment = variants[1:]
        
        for var in treatment:
            stat, p_val = stats.ttest_ind_from_stats(
                variant_results[control].mean,
                variant_results[control].std,
                variant_results[control].sample_size,
                variant_results[var].mean,
                variant_results[var].std,
                variant_results[var].sample_size
            )
            
            if p_val > (1 - confidence_level):
                return False
                
        return True
    
    def _generate_visualization(self, results: Dict) -> Dict:
        """Generate visualization of experiment results."""
        figs = {}
        for metric, metric_results in results.items():
            fig = make_subplots(rows=2, cols=1)
            
            # Add mean values
            means = []
            cis = []
            variants = []
            
            for variant, stats in metric_results["variant_results"].items():
                means.append(stats.mean)
                cis.append([stats.confidence_interval[0], stats.confidence_interval[1]])
                variants.append(variant)
            
            # Bar plot for means
            fig.add_trace(
                go.Bar(x=variants, y=means, name="Mean"),
                row=1, col=1
            )
            
            # Error bars for confidence intervals
            fig.add_trace(
                go.Scatter(
                    x=variants,
                    y=[ci[1] - ci[0] for ci in cis],
                    name="Confidence Interval",
                    error_y=dict(type="data", array=[ci[1] - ci[0] for ci in cis])
                ),
                row=2, col=1
            )
            
            figs[metric] = fig.to_json()
            
        return figs
    
    def _log_experiment_event(
        self,
        experiment_id: str,
        event_type: str,
        data: Dict
    ):
        """Log experiment events to database."""
        with self.db.get_session() as session:
            session.execute(
                """
                INSERT INTO experiment_events
                (experiment_id, event_type, event_data, created_at)
                VALUES (:experiment_id, :event_type, :event_data, :created_at)
                """,
                {
                    "experiment_id": experiment_id,
                    "event_type": event_type,
                    "event_data": json.dumps(data),
                    "created_at": datetime.now()
                }
            )
    
    def _update_statistical_significance(self, experiment_id: str):
        """Update experiment's statistical significance status."""
        exp = self.active_experiments[experiment_id]
        
        # Check sample size requirements
        for variant_results in exp["results"].values():
            if len(variant_results) < exp["config"].min_sample_size:
                return
                
        # Check duration
        duration = datetime.now() - exp["start_time"]
        if duration.days >= exp["config"].max_duration_days:
            exp["status"] = "completed"
            
        # Check significance for each metric
        for metric in exp["config"].metrics:
            results = self.get_experiment_results(experiment_id, metric)
            if results["results"][metric]["significant_difference"]:
                exp["statistical_significance"] = True
                if exp["status"] != "completed":
                    exp["status"] = "significant"
