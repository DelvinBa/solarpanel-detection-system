#!/usr/bin/env python
# Simplified YOLO Solar Panel Detection Training Script with MLflow Integration

import os
import mlflow
import logging
import traceback
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_mlflow():
    """Configure MLflow tracking - simplified version"""
    try:
        print("Setting up MLflow...")
        # Set up the tracking URI to use the MLflow tracking server
        tracking_uri = "http://localhost:5001"
        print(f"Setting MLflow tracking URI: {tracking_uri}")
        mlflow.set_tracking_uri(tracking_uri)
        
        # Create experiment
        experiment_name = "solar_panel_detection_simple"
        print(f"Creating experiment: {experiment_name}")
        
        # Check if experiment exists
        print("Checking if experiment exists...")
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"Experiment does not exist, creating new one: {experiment_name}")
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"Created experiment with ID: {experiment_id}")
        else:
            experiment_id = experiment.experiment_id
            print(f"Using existing experiment with ID: {experiment_id}")
        
        # Set as active experiment
        print(f"Setting active experiment to: {experiment_name}")
        mlflow.set_experiment(experiment_name)
        
        return experiment_id
    except Exception as e:
        print(f"Error setting up MLflow: {str(e)}")
        print(traceback.format_exc())
        raise

def run_simple_training_log():
    """Just log some metrics without actual training"""
    try:
        # Setup MLflow
        print("Starting MLflow setup...")
        experiment_id = setup_mlflow()
        
        # Start MLflow run
        print(f"Starting MLflow run with experiment ID: {experiment_id}")
        with mlflow.start_run(experiment_id=experiment_id) as run:
            run_id = run.info.run_id
            print(f"Started MLflow run with ID: {run_id}")
            
            # Log fake parameters
            print("Logging parameters...")
            params = {
                "model": "yolov8n.pt",
                "epochs": 50,
                "batch_size": 16,
                "img_size": 832,
            }
            mlflow.log_params(params)
            print(f"Logged parameters: {params}")
            
            # Log fake metrics
            print("Logging metrics...")
            metrics = {
                "mAP50": 0.85,
                "mAP50-95": 0.65,
                "precision": 0.88,
                "recall": 0.82,
            }
            mlflow.log_metrics(metrics)
            print(f"Logged metrics: {metrics}")
            
            # Create a dummy artifact
            print("Creating dummy artifact...")
            models_dir = Path("../../models/dummy")
            models_dir.mkdir(parents=True, exist_ok=True)
            dummy_file = models_dir / "dummy.txt"
            with open(dummy_file, "w") as f:
                f.write("This is a dummy model file")
            
            # Log the dummy artifact
            print(f"Logging artifact from: {dummy_file}")
            mlflow.log_artifact(str(dummy_file), "model")
            print(f"Logged dummy artifact")
            
            print(f"Training completed. MLflow run ID: {run_id}")
            print(f"View run details at: http://localhost:5001/experiments")
    except Exception as e:
        print(f"Error in run_simple_training_log: {str(e)}")
        print(traceback.format_exc())

if __name__ == "__main__":
    print("Starting script...")
    run_simple_training_log()
    print("Script completed.") 