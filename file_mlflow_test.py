import mlflow
import os
import time

# Set the tracking URI to the local file system
tracking_uri = "file:./mlruns"
print(f"Setting MLflow tracking URI: {tracking_uri}")
mlflow.set_tracking_uri(tracking_uri)

# Create a new experiment
experiment_name = "file_based_experiment"
print(f"Creating experiment: {experiment_name}")

try:
    # Try to get existing experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment:
        experiment_id = experiment.experiment_id
        print(f"Experiment already exists with ID: {experiment_id}")
    else:
        # Create the experiment
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Created experiment with ID: {experiment_id}")
    
    # Start a run
    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id
        print(f"Started run with ID: {run_id}")
        
        # Log a parameter
        mlflow.log_param("model", "yolov8n.pt")
        mlflow.log_param("epochs", 50)
        
        # Log a metric
        mlflow.log_metric("mAP50", 0.85)
        mlflow.log_metric("precision", 0.78)
        
        print("Run completed successfully")
        print(f"Data stored in: {os.path.abspath('./mlruns')}")
        
except Exception as e:
    print(f"Error: {str(e)}") 