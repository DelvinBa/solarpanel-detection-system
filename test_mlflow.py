import mlflow
import random
import os

# Set MLflow tracking URI to a local directory
mlflow.set_tracking_uri("file:./mlruns")

# Create or get an experiment
experiment_name = "test_experiment"
experiment_id = mlflow.set_experiment(experiment_name).experiment_id
print(f"Using experiment '{experiment_name}' with ID: {experiment_id}")

# Start an MLflow run
with mlflow.start_run() as run:
    # Log parameters
    mlflow.log_param("param1", 5)
    mlflow.log_param("param2", "test")
    
    # Log metrics
    mlflow.log_metric("accuracy", 0.85)
    mlflow.log_metric("loss", 0.15)
    
    # Log a sequence of metrics
    for i in range(10):
        mlflow.log_metric("iteration_metric", random.random(), step=i)
    
    # Get the run ID for verification
    run_id = run.info.run_id
    print(f"MLflow run completed. Run ID: {run_id}")
    print(f"View the run at http://localhost:5000/#/experiments/{experiment_id}/runs/{run_id}") 