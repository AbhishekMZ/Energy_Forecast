"""FastAPI backend for the Energy Forecasting System."""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import pandas as pd
import json
import logging
from pathlib import Path
import asyncio
from datetime import datetime
import uuid
from auth import routes as auth_routes
from auth.security import get_current_active_user, User
from models.database import get_db, Base, engine
from models.integrated_pipeline import IntegratedPipeline

# Create database tables
Base.metadata.create_all(bind=engine)

# Initialize FastAPI app
app = FastAPI(
    title="Energy Forecast API",
    description="API for energy consumption forecasting system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include authentication routes
app.include_router(auth_routes.router, prefix="/api", tags=["authentication"])

# Initialize pipeline
pipeline = IntegratedPipeline('config/default_config.json')

# Store active jobs
active_jobs: Dict[str, Dict[str, Any]] = {}

class TrainingRequest(BaseModel):
    """Training request model."""
    model_type: str
    optimize_config: bool = True
    config_overrides: Optional[Dict[str, Any]] = None

class PredictionRequest(BaseModel):
    """Prediction request model."""
    model_id: str
    data: List[Dict[str, Any]]

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Energy Forecast API",
        "version": "1.0.0",
        "status": "active"
    }

@app.get("/health")
async def health_check(db: Session = Depends(get_db)):
    try:
        # Verify database connection
        db.execute("SELECT 1")
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
    
    return {
        "status": "ok",
        "database": db_status
    }

@app.get("/models")
async def get_available_models(current_user: User = Depends(get_current_active_user)):
    """Get list of available models."""
    return {
        "models": pipeline.get_available_models()
    }

@app.get("/models/{model_type}")
async def get_model_info(model_type: str, current_user: User = Depends(get_current_active_user)):
    """Get information about specific model."""
    try:
        return pipeline.get_model_info(model_type)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

def process_training(
    job_id: str,
    data_path: str,
    model_type: str,
    optimize_config: bool,
    config_overrides: Optional[Dict[str, Any]] = None
):
    """Process training job."""
    try:
        # Update job status
        active_jobs[job_id]["status"] = "running"
        
        # Run pipeline
        results = pipeline.run_pipeline(
            data_path=data_path,
            target_col="consumption",  # TODO: Make configurable
            model_type=model_type,
            optimize_config=optimize_config
        )
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path('output') / job_id
        pipeline.save_results(results, str(output_dir))
        
        # Update job status
        active_jobs[job_id].update({
            "status": "completed",
            "results": {
                "metrics": results["metrics"],
                "output_dir": str(output_dir)
            }
        })
        
    except Exception as e:
        # Update job status with error
        active_jobs[job_id].update({
            "status": "failed",
            "error": str(e)
        })
        logging.error(f"Training failed for job {job_id}: {str(e)}")

@app.post("/train")
async def train_model(
    background_tasks: BackgroundTasks,
    request: TrainingRequest,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user)
):
    """Train a model with uploaded data."""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    try:
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Save uploaded file
        data_path = Path('data') / f'upload_{job_id}.csv'
        data_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(data_path, 'wb') as f:
            content = await file.read()
            f.write(content)
        
        # Initialize job
        active_jobs[job_id] = {
            "id": job_id,
            "status": "initialized",
            "model_type": request.model_type,
            "timestamp": datetime.now().isoformat()
        }
        
        # Start training in background
        background_tasks.add_task(
            process_training,
            job_id,
            str(data_path),
            request.model_type,
            request.optimize_config,
            request.config_overrides
        )
        
        return {
            "job_id": job_id,
            "status": "initialized"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str, current_user: User = Depends(get_current_active_user)):
    """Get status of training job."""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return active_jobs[job_id]

@app.post("/predict/{model_id}")
async def predict(
    model_id: str,
    request: PredictionRequest,
    current_user: User = Depends(get_current_active_user)
):
    """Make predictions using trained model."""
    try:
        # Load model
        model_path = Path('output') / model_id / 'model.pkl'
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="Model not found")
        
        model = pipeline.load_model(str(model_path))
        
        # Convert input data to DataFrame
        data = pd.DataFrame(request.data)
        
        # Make predictions
        predictions = model.predict(data)
        
        return {
            "predictions": predictions.tolist()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/performance")
async def get_performance_metrics(current_user: User = Depends(get_current_active_user)):
    """Get system performance metrics."""
    try:
        return pipeline.training_pipeline.get_performance_report()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/monitoring/models")
async def get_model_metrics(current_user: User = Depends(get_current_active_user)):
    """Get model metrics."""
    return {
        "model_metrics": pipeline.get_model_metrics(),
        "training_history": pipeline.get_training_history(),
        "prediction_stats": pipeline.get_prediction_stats(),
        "resource_usage": pipeline.get_resource_usage()
    }

@app.get("/monitoring/errors")
async def get_error_metrics(current_user: User = Depends(get_current_active_user)):
    """Get error metrics."""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return pipeline.get_error_metrics()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
