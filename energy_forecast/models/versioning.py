"""Model versioning and A/B testing implementation."""

import mlflow
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ModelVersioning:
    def __init__(self):
        self.mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        self.experiment_name = "energy_forecast"
        
    def log_model_version(
        self,
        model,
        version: str,
        metrics: Dict[str, float],
        params: Dict[str, Any]
    ) -> str:
        """Log a new model version with metrics and parameters."""
        try:
            with mlflow.start_run() as run:
                # Log model parameters
                mlflow.log_params(params)
                
                # Log model metrics
                mlflow.log_metrics(metrics)
                
                # Save model
                mlflow.sklearn.log_model(model, "model")
                
                # Log version as tag
                mlflow.set_tag("version", version)
                
                return run.info.run_id
        except Exception as e:
            logger.error(f"Failed to log model version: {str(e)}")
            raise

    def get_model_version(self, version: str) -> Optional[Dict]:
        """Retrieve a specific model version."""
        try:
            client = mlflow.tracking.MlflowClient()
            runs = client.search_runs(
                experiment_ids=[mlflow.get_experiment_by_name(self.experiment_name).experiment_id],
                filter_string=f"tags.version = '{version}'"
            )
            if runs:
                return {
                    "run_id": runs[0].info.run_id,
                    "metrics": runs[0].data.metrics,
                    "params": runs[0].data.params,
                    "version": version
                }
            return None
        except Exception as e:
            logger.error(f"Failed to get model version: {str(e)}")
            raise

    def compare_versions(self, version1: str, version2: str) -> Dict:
        """Compare metrics between two model versions."""
        v1 = self.get_model_version(version1)
        v2 = self.get_model_version(version2)
        
        if not v1 or not v2:
            raise ValueError("One or both versions not found")
            
        return {
            "metrics_diff": {
                k: v2["metrics"][k] - v1["metrics"][k]
                for k in v1["metrics"].keys()
                if k in v2["metrics"]
            },
            "version1": v1,
            "version2": v2
        }

class ABTesting:
    def __init__(self):
        self.active_experiments = {}
        
    def create_experiment(
        self,
        name: str,
        variants: Dict[str, float],
        metadata: Dict[str, Any]
    ) -> str:
        """Create a new A/B testing experiment."""
        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.active_experiments[experiment_id] = {
            "name": name,
            "variants": variants,
            "metadata": metadata,
            "results": {variant: [] for variant in variants.keys()},
            "start_time": datetime.now()
        }
        return experiment_id
        
    def log_variant_result(
        self,
        experiment_id: str,
        variant: str,
        metrics: Dict[str, float]
    ):
        """Log results for a specific variant in an experiment."""
        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
            
        if variant not in self.active_experiments[experiment_id]["variants"]:
            raise ValueError(f"Variant {variant} not found in experiment")
            
        self.active_experiments[experiment_id]["results"][variant].append({
            "metrics": metrics,
            "timestamp": datetime.now()
        })
        
    def get_experiment_results(self, experiment_id: str) -> Dict:
        """Get results for a specific experiment."""
        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
            
        exp = self.active_experiments[experiment_id]
        results = {}
        
        for variant, data in exp["results"].items():
            if data:
                metrics = {
                    k: sum(d["metrics"][k] for d in data) / len(data)
                    for k in data[0]["metrics"].keys()
                }
                results[variant] = metrics
                
        return {
            "experiment_name": exp["name"],
            "variants": exp["variants"],
            "results": results,
            "duration": (datetime.now() - exp["start_time"]).total_seconds()
        }
