"""
YOLO Model Deployment DAG
=========================

This DAG handles the deployment of YOLO models that have been registered
to the Production stage in the MLflow Model Registry.

The pipeline includes:
1. Checking for production models
2. Downloading the model from MLflow
3. Converting to ONNX format for faster inference
4. Copying to the deployment directory
5. Updating the running service

"""

from datetime import datetime, timedelta
import os
import shutil
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.python import PythonSensor
from airflow.models import Variable
from airflow.utils.task_group import TaskGroup

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'deploy_yolo_model',
    default_args=default_args,
    description='Deploy YOLO model from MLflow to the inferencing service',
    schedule_interval='0 0 * * *',  # Run daily at midnight
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['yolo', 'mlflow', 'solar-panel', 'deployment', 'production'],
)

# MLflow configuration
MLFLOW_TRACKING_URI = f"http://mlflow:5000"
MODEL_NAME = "solar_panel_yolo"
DEPLOYMENT_DIR = '/opt/airflow/dags/models/production'

def check_for_production_model(**kwargs):
    """
    Check if there are models in Production stage that haven't been deployed yet
    """
    import mlflow
    from mlflow.tracking import MlflowClient
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    
    # Get latest production model version
    production_versions = [
        mv for mv in client.search_model_versions(f"name='{MODEL_NAME}'")
        if mv.current_stage == "Production"
    ]
    
    if not production_versions:
        print("No production models found")
        return False
    
    # Sort by version number (descending)
    production_versions.sort(key=lambda x: int(x.version), reverse=True)
    latest_prod_version = production_versions[0]
    
    # Check if this version has been deployed already
    deployed_version_file = Path(DEPLOYMENT_DIR) / "version.txt"
    
    if deployed_version_file.exists():
        with open(deployed_version_file, "r") as f:
            deployed_version = f.read().strip()
        
        if deployed_version == latest_prod_version.version:
            print(f"Latest production version {latest_prod_version.version} already deployed")
            return False
    
    # Store the latest production model information
    kwargs['ti'].xcom_push(key='model_version', value=latest_prod_version.version)
    kwargs['ti'].xcom_push(key='run_id', value=latest_prod_version.run_id)
    print(f"Found new production model version {latest_prod_version.version} (run_id: {latest_prod_version.run_id})")
    return True

def download_model_from_mlflow(**kwargs):
    """
    Download the model artifacts from MLflow
    """
    import mlflow
    import os
    from pathlib import Path
    
    ti = kwargs['ti']
    run_id = ti.xcom_pull(task_ids='check_for_production_model', key='run_id')
    version = ti.xcom_pull(task_ids='check_for_production_model', key='model_version')
    
    if not run_id:
        raise ValueError("No run_id provided")
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Create a temporary directory for downloads
    temp_dir = Path(f"/tmp/model_download_{run_id}")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Download the best model
    artifact_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="best_model/best.pt",
        dst_path=str(temp_dir)
    )
    
    model_path = Path(artifact_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Downloaded model not found at {model_path}")
    
    print(f"Model downloaded to {model_path}")
    
    # Pass the model path to the next tasks
    kwargs['ti'].xcom_push(key='model_path', value=str(model_path))
    return str(model_path)

def convert_to_onnx(**kwargs):
    """
    Convert PyTorch model to ONNX format for faster inference
    """
    from ultralytics import YOLO
    import os
    from pathlib import Path
    
    ti = kwargs['ti']
    model_path = ti.xcom_pull(task_ids='download_model', key='model_path')
    
    if not model_path or not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    # Load the model
    model = YOLO(model_path)
    
    # Define the output directory
    output_dir = Path(model_path).parent / "onnx"
    os.makedirs(output_dir, exist_ok=True)
    
    # Export to ONNX
    onnx_path = output_dir / "model.onnx"
    success = model.export(format="onnx", imgsz=832)
    
    # The export method returns the path to the exported model
    onnx_path = Path(success)
    
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found at {onnx_path}")
    
    print(f"Model converted to ONNX format: {onnx_path}")
    
    # Pass the ONNX model path to the next tasks
    kwargs['ti'].xcom_push(key='onnx_path', value=str(onnx_path))
    return str(onnx_path)

def deploy_model(**kwargs):
    """
    Copy the model files to the deployment directory
    """
    import os
    import shutil
    from pathlib import Path
    
    ti = kwargs['ti']
    model_path = ti.xcom_pull(task_ids='download_model', key='model_path')
    onnx_path = ti.xcom_pull(task_ids='convert_to_onnx', key='onnx_path')
    version = ti.xcom_pull(task_ids='check_for_production_model', key='model_version')
    
    if not model_path or not onnx_path:
        raise ValueError("Missing model paths")
    
    # Create deployment directory if it doesn't exist
    deploy_dir = Path(DEPLOYMENT_DIR)
    os.makedirs(deploy_dir, exist_ok=True)
    
    # Copy the PyTorch model
    pt_dest = deploy_dir / "model.pt"
    shutil.copy2(model_path, pt_dest)
    
    # Copy the ONNX model
    onnx_dest = deploy_dir / "model.onnx"
    shutil.copy2(onnx_path, onnx_dest)
    
    # Write version info
    with open(deploy_dir / "version.txt", "w") as f:
        f.write(str(version))
    
    # Write deployment timestamp
    with open(deploy_dir / "deployed_at.txt", "w") as f:
        from datetime import datetime
        f.write(datetime.now().isoformat())
    
    print(f"Models deployed to {deploy_dir}")
    print(f"PyTorch model: {pt_dest}")
    print(f"ONNX model: {onnx_dest}")
    print(f"Deployed version: {version}")
    
    return {
        "deploy_dir": str(deploy_dir),
        "pt_model": str(pt_dest),
        "onnx_model": str(onnx_dest),
        "version": version
    }

def restart_inference_service(**kwargs):
    """
    Restart the inference service to use the new model
    """
    import subprocess
    import time
    
    # This would typically send a signal to restart your inference service
    # For this example, we'll just simulate it with a sleep
    print("Sending restart signal to inference service...")
    time.sleep(2)
    
    # Check if the service is responsive after restart
    max_retries = 5
    retry_interval = 5
    
    for i in range(max_retries):
        try:
            # Here you would make a request to your inference service to check if it's alive
            # For example:
            response = requests.get("http://solarpanel_detection_service:8000/health")
            if response.status_code == 200:
                print("Service is responsive!")
                break
            
            # Simulating success
            print(f"Retry {i+1}/{max_retries}: Service started successfully")
            return True
        except Exception as e:
            print(f"Retry {i+1}/{max_retries}: Service not responsive yet. Error: {e}")
            time.sleep(retry_interval)
    
    raise Exception("Failed to restart inference service after multiple retries")

# Define the tasks in the DAG
with dag:
    # Task to check if there's a new production model
    check_production_model_sensor = PythonSensor(
        task_id='check_for_production_model',
        python_callable=check_for_production_model,
        mode='reschedule',  # This means the sensor will be rescheduled after each check
        poke_interval=600,  # Check every 10 minutes
        timeout=60 * 60 * 2,  # Timeout after 2 hours
        retries=0,  # No retries for the sensor itself
    )
    
    # Task to download the model from MLflow
    download_model_task = PythonOperator(
        task_id='download_model',
        python_callable=download_model_from_mlflow,
        provide_context=True,
    )
    
    # Task to convert to ONNX
    convert_to_onnx_task = PythonOperator(
        task_id='convert_to_onnx',
        python_callable=convert_to_onnx,
        provide_context=True,
    )
    
    # Task to deploy the model
    deploy_model_task = PythonOperator(
        task_id='deploy_model',
        python_callable=deploy_model,
        provide_context=True,
    )
    
    # Task to restart the inference service
    restart_service_task = PythonOperator(
        task_id='restart_inference_service',
        python_callable=restart_inference_service,
        provide_context=True,
    )
    
    # Define task dependencies
    check_production_model_sensor >> download_model_task >> convert_to_onnx_task >> deploy_model_task >> restart_service_task 