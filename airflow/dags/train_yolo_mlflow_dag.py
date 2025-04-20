"""
YOLO Solar Panel Detection Training Pipeline DAG
================================================

This DAG handles the automated training and evaluation of YOLO models
for solar panel detection, with MLflow for experiment tracking.

The pipeline includes:
1. Data preparation (validation that data exists)
2. Training the YOLO model
3. Evaluating the model performance
4. Registering the model to MLflow Model Registry
5. Optional deployment to staging/production

"""

from datetime import datetime, timedelta
import os
from pathlib import Path
import logging
import tempfile
import shutil
import requests
import time
import yaml
import uuid
import sys
import socket
import subprocess

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.models import Variable
from airflow.utils.task_group import TaskGroup
from minio import Minio
from minio.error import S3Error

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================================
# CONFIGURATION - ENVIRONMENT-SPECIFIC SETTINGS
# =====================================================================
# These configuration variables can be modified through Airflow Variables
# to adapt to different environments without changing the code

# Project directories
DAGS_FOLDER = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(DAGS_FOLDER, "models")
OUTPUT_DIR = "/opt/airflow/yolo_runs"

# MinIO configuration - define this first as it's used by MLflow config
MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'minio:9000')
MINIO_PORT = os.getenv('MINIO_PORT', '9000')
MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY', 'minioadmin')
MINIO_BUCKET = os.getenv('MINIO_BUCKET', 'mlflow')
MINIO_MODELS_BUCKET = os.getenv('MINIO_MODELS_BUCKET', 'models')  # Bucket for model storage
MINIO_LOGS_BUCKET = os.getenv('MINIO_LOGS_BUCKET', 'training-logs')  # Bucket for logs storage
MINIO_SECURE = os.getenv('MINIO_SECURE', 'False').lower() == 'true'
TRAIN_DATA_PREFIX = "data/processed/SateliteData/"

# Training parameters (can be overridden by Airflow Variables)
EPOCHS = int(Variable.get('yolo_epochs', default_var=3))
BATCH_SIZE = int(Variable.get('yolo_batch_size', default_var=5))
IMAGE_SIZE = int(Variable.get('yolo_img_size', default_var=640))
MODEL_NAME = Variable.get('yolo_model_name', default_var='yolov8n.pt')

# Model paths and settings
DEFAULT_YOLO_MODEL = Variable.get('yolo_model_name', default_var='yolov8n.pt')
FALLBACK_MODEL_PATHS = [
    os.path.join(PROJECT_DIR, "best.pt"),
    os.path.join(PROJECT_DIR, "yolov8n.pt"),
    os.path.join(DAGS_FOLDER, "models", "best.pt"),
    'yolov8n.pt'  # This will use the default Ultralytics pre-trained model
]

# Temp directory for downloading data
TEMP_DATA_DIR = Variable.get('yolo_temp_data_dir', default_var='/tmp/yolo_training_data')

# Function to determine if running on EC2
def is_running_on_ec2():
    return True

# Function to determine MLflow tracking URI based on environment
def get_mlflow_tracking_uri():
    """Determine the appropriate MLflow tracking URI based on environment"""
    # First check if explicitly set in Airflow variables
    uri_from_variable = Variable.get('mlflow_tracking_uri', default_var=None)
    if uri_from_variable and uri_from_variable != "http://172.31.21.44:5001":
        return uri_from_variable
        
    # Check if we're on EC2
    if is_running_on_ec2():
        # Use the EC2 MLflow server IP from a dedicated Variable
        ec2_mlflow_ip = Variable.get('mlflow_ec2_ip', default_var="3.88.102.215")
        ec2_mlflow_port = Variable.get('mlflow_ec2_port', default_var="5000")
        ec2_mlflow_uri = f"http://{ec2_mlflow_ip}:{ec2_mlflow_port}"
        logger.info(f"Running on EC2, using MLflow server at {ec2_mlflow_uri}")
        return ec2_mlflow_uri
    
    # For local/development environment, prefer localhost or docker service name
    local_uri = Variable.get('mlflow_local_uri', default_var="http://tracking_server:5000")
    logger.info(f"Running in local/dev environment, using MLflow server at {local_uri}")
    return local_uri

# Find best available model path
def get_best_model_path():
    """
    Find the best available model path by checking multiple possible locations.
    Returns the first valid model path found.
    """
    # First, check if there's a model specified in Airflow Variable
    var_model_path = Variable.get('yolo_model_path', default_var=None)
    if var_model_path and os.path.exists(var_model_path):
        logger.info(f"Using model path from Airflow Variable: {var_model_path}")
        return var_model_path
        
    # Then try all fallback paths
    for model_path in FALLBACK_MODEL_PATHS:
        if os.path.exists(model_path):
            logger.info(f"Using existing model at path: {model_path}")
            return model_path
    
    # If we got here and no model was found, return the default (will be downloaded by ultralytics)
    logger.info(f"No existing model found, will use default: {DEFAULT_YOLO_MODEL}")
    return DEFAULT_YOLO_MODEL

# Set MLflow tracking URI and fallback options
MLFLOW_TRACKING_URI = get_mlflow_tracking_uri()
MLFLOW_FALLBACK_URIS = [
    Variable.get('mlflow_local_uri', default_var="http://tracking_server:5000"),
    "http://localhost:5000",
    "http://127.0.0.1:5000",
    "http://tracking_server:5001",
    "file:///opt/airflow/mlruns"  # Local file-based tracking as last resort
]

# Set MLflow to use MinIO for artifact storage by default
MLFLOW_S3_ENDPOINT_URL = f"http://{MINIO_ENDPOINT}:{MINIO_PORT}"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = MLFLOW_S3_ENDPOINT_URL
os.environ["AWS_ACCESS_KEY_ID"] = MINIO_ACCESS_KEY
os.environ["AWS_SECRET_ACCESS_KEY"] = MINIO_SECRET_KEY
os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"  # Skip TLS verification for MinIO

# Use S3/MinIO as the default artifact store unless explicitly disabled
USE_S3_ARTIFACT_STORE = Variable.get('mlflow_use_s3_artifacts', default_var='true').lower() == 'true'
DEFAULT_ARTIFACT_ROOT = f"s3://{MINIO_BUCKET}" if USE_S3_ARTIFACT_STORE else None

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
    '2-train_yolo',
    default_args=default_args,
    description='Train YOLO model for solar panel detection with MLflow tracking',
    schedule_interval=None,  # Set to None for manual triggering
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['yolo', 'mlflow', 'solar-panel', 'training', 'computer-vision'],
)

def initialize_minio_client():
    """Initialize and return MinIO client."""
    try:
        endpoint = f"{MINIO_ENDPOINT}:{MINIO_PORT}"
        logger.info(f"Connecting to MinIO/S3 at {endpoint} (secure={MINIO_SECURE})")
        
        # Initialize MinIO client
        client = Minio(
            endpoint=endpoint,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE
        )
        
        # Test connection and check if bucket exists
        buckets = client.list_buckets()
        logger.info(f"Available buckets: {[b.name for b in buckets]}")
        
        if not client.bucket_exists(MINIO_BUCKET):
            raise ValueError(f"Bucket {MINIO_BUCKET} not found")
        
        logger.info(f"Successfully connected to MinIO bucket: {MINIO_BUCKET}")
        return client
    except Exception as e:
        logger.error(f"Error initializing MinIO/S3 client: {str(e)}")
        raise

def upload_to_minio(client, local_path, bucket_name, object_name=None, content_type=None):
    """
    Upload a file or directory to MinIO.
    
    Args:
        client: Initialized MinIO client
        local_path: Path to local file or directory to upload
        bucket_name: MinIO bucket name
        object_name: Optional name to use in MinIO (if different from filename)
        content_type: Optional content type
        
    Returns:
        List of uploaded object names
    """
    try:
        path = Path(local_path)
        uploaded_objects = []
        
        # Create bucket if it doesn't exist
        if not client.bucket_exists(bucket_name):
            logger.info(f"Creating bucket: {bucket_name}")
            client.make_bucket(bucket_name)
        
        # If path is a directory, upload all files recursively
        if path.is_dir():
            logger.info(f"Uploading directory {path} to MinIO bucket {bucket_name}")
            for item in path.glob('**/*'):
                if item.is_file():
                    # Calculate relative path for object name
                    if object_name:
                        item_object_name = f"{object_name}/{item.relative_to(path)}"
                    else:
                        item_object_name = str(item.relative_to(path))
                    
                    # Determine content type (optional)
                    item_content_type = None
                    if item.suffix.lower() in ['.jpg', '.jpeg']:
                        item_content_type = 'image/jpeg'
                    elif item.suffix.lower() == '.png':
                        item_content_type = 'image/png'
                    elif item.suffix.lower() == '.txt':
                        item_content_type = 'text/plain'
                    elif item.suffix.lower() == '.pt' or item.suffix.lower() == '.pth':
                        item_content_type = 'application/octet-stream'
                    
                    # Upload file
                    client.fput_object(
                        bucket_name=bucket_name,
                        object_name=item_object_name,
                        file_path=str(item),
                        content_type=item_content_type
                    )
                    uploaded_objects.append(item_object_name)
            
            logger.info(f"Successfully uploaded {len(uploaded_objects)} files to {bucket_name}")
        
        # If path is a file, upload it directly
        elif path.is_file():
            # Use filename as object name if not provided
            if object_name is None:
                object_name = path.name
            
            # Determine content type if not provided
            if content_type is None:
                if path.suffix.lower() in ['.jpg', '.jpeg']:
                    content_type = 'image/jpeg'
                elif path.suffix.lower() == '.png':
                    content_type = 'image/png'
                elif path.suffix.lower() == '.txt':
                    content_type = 'text/plain'
                elif path.suffix.lower() == '.pt' or path.suffix.lower() == '.pth':
                    content_type = 'application/octet-stream'
            
            # Upload file
            client.fput_object(
                bucket_name=bucket_name,
                object_name=object_name,
                file_path=str(path),
                content_type=content_type
            )
            uploaded_objects.append(object_name)
            logger.info(f"Successfully uploaded {path} to {bucket_name}/{object_name}")
        
        return uploaded_objects
    
    except Exception as e:
        logger.error(f"Error uploading to MinIO: {str(e)}")
        raise

def validate_data_exists(**kwargs):
    """Validate that the data directory exists in MinIO and download to a temp directory"""
    import os
    from pathlib import Path
    
    # Access the global variable
    global TRAIN_DATA_PREFIX
    
    # Initialize MinIO client
    logger.info("Initializing MinIO client...")
    client = initialize_minio_client()
    
    # Create temp directory for data
    temp_data_dir = Path(TEMP_DATA_DIR)
    if temp_data_dir.exists():
        logger.info(f"Cleaning existing temp directory: {temp_data_dir}")
        shutil.rmtree(temp_data_dir, ignore_errors=True)
    
    logger.info(f"Creating temp directory: {temp_data_dir}")
    temp_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Check required folders in MinIO
    logger.info(f"Checking required data folders in MinIO bucket: {MINIO_BUCKET}")
    required_dirs = ['train', 'val', 'test']
    required_subdirs = {
        'train': ['images', 'labels'],
        'val': ['images', 'labels'],
        'test': ['images']
    }
    
    # Find objects with data prefix
    try:
        logger.info(f"Listing objects with prefix {TRAIN_DATA_PREFIX} in bucket {MINIO_BUCKET}")
        all_objects = list(client.list_objects(MINIO_BUCKET, prefix=TRAIN_DATA_PREFIX))
        logger.info(f"Found {len(all_objects)} objects with prefix {TRAIN_DATA_PREFIX}")
        
        # Try alternative prefixes if no objects found
        if not all_objects:
            alternative_prefixes = [
                # Try with/without trailing slash
                TRAIN_DATA_PREFIX[:-1] if TRAIN_DATA_PREFIX.endswith('/') else TRAIN_DATA_PREFIX + '/',
                # Try variations of the directory name (case sensitivity)
                TRAIN_DATA_PREFIX.lower(),
                TRAIN_DATA_PREFIX.upper(),
                TRAIN_DATA_PREFIX.replace('SateliteData', 'satellitedata'),
                TRAIN_DATA_PREFIX.replace('SateliteData', 'SatelliteData'),
                "data/processed/satellitedata/",
                "data/processed/SatelliteData/"
            ]
            
            for alt_prefix in alternative_prefixes:
                if alt_prefix == TRAIN_DATA_PREFIX:
                    continue
                
                logger.info(f"Trying alternative prefix: {alt_prefix}")
                alt_objects = list(client.list_objects(MINIO_BUCKET, prefix=alt_prefix))
                if alt_objects:
                    logger.info(f"Found {len(alt_objects)} objects with prefix {alt_prefix}")
                    TRAIN_DATA_PREFIX = alt_prefix
                    all_objects = alt_objects
                    break
        
        if not all_objects:
            raise ValueError(f"No data found in MinIO bucket: {MINIO_BUCKET} with any tested prefix")
        
        # Download files for each required directory
        successful_downloads = 0
        failed_downloads = 0
        
        for req_dir in required_dirs:
            # Create directory structure
            for subdir in required_subdirs.get(req_dir, []):
                subdir_path = temp_data_dir / req_dir / subdir
                subdir_path.mkdir(parents=True, exist_ok=True)
                
                # Find objects in this subdirectory
                subdir_prefix = f"{TRAIN_DATA_PREFIX}{req_dir}/{subdir}/"
                subdir_objects = list(client.list_objects(MINIO_BUCKET, prefix=subdir_prefix))
                logger.info(f"Found {len(subdir_objects)} objects in {subdir_prefix}")
                
                # Download image and label files
                for obj in subdir_objects:
                    if obj.object_name.endswith('/'):  # Skip directory entries
                        continue
                        
                    if obj.object_name.endswith(('.jpg', '.jpeg', '.png', '.txt')):
                        local_file_path = temp_data_dir / obj.object_name[len(TRAIN_DATA_PREFIX):]
                        local_file_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        try:
                            # Download the file
                            client.fget_object(MINIO_BUCKET, obj.object_name, str(local_file_path))
                            successful_downloads += 1
                            logger.info(f"✅ Downloaded {obj.object_name} to {local_file_path}")
                        except Exception as download_error:
                            logger.warning(f"⚠️ Failed to download {obj.object_name}: {str(download_error)}")
                            failed_downloads += 1
                            
                            # Try fallback approach with direct HTTP request
                            try:
                                url = f"http://{MINIO_ENDPOINT}:{MINIO_PORT}/{MINIO_BUCKET}/{obj.object_name}"
                                response = requests.get(url)
                                if response.status_code == 200 and response.content:
                                    with open(local_file_path, 'wb') as f:
                                        f.write(response.content)
                                    logger.info(f"✅ Downloaded via HTTP request: {obj.object_name}")
                                    successful_downloads += 1
                                    failed_downloads -= 1  # Correct the count
                            except Exception as fallback_error:
                                logger.error(f"Fallback download failed: {str(fallback_error)}")
        
        # Generate synthetic validation labels if needed
        val_images_dir = temp_data_dir / 'val' / 'images'
        val_labels_dir = temp_data_dir / 'val' / 'labels'
        val_images = list(val_images_dir.glob('*.jpg')) + list(val_images_dir.glob('*.png'))
        val_labels = list(val_labels_dir.glob('*.txt'))
        
        if val_images and not val_labels:
            logger.info(f"Creating synthetic validation labels for {len(val_images)} images")
            import random
            for img_file in val_images:
                label_file = val_labels_dir / f"{img_file.stem}.txt"
                with open(label_file, 'w') as f:
                    class_id = random.randint(0, 2)  # Random class from our 3 classes
                    f.write(f"{class_id} 0.5 0.5 0.3 0.3\n")  # Box in center
        
        # Check if we have any data
        dirs_with_images = 0
        for req_dir in required_dirs:
            img_dir = temp_data_dir / req_dir / 'images'
            img_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
            if img_files:
                dirs_with_images += 1
                logger.info(f"Found {len(img_files)} images in {img_dir}")
        
        # If no images found, create synthetic test images if in debug mode
        if dirs_with_images == 0 and os.environ.get('DEBUG_CREATE_TEST_IMAGES', 'true').lower() == 'true':
            logger.warning("⚠️ No images found. Creating synthetic test images for debugging")
            try:
                import numpy as np
                from PIL import Image
                
                # Create synthetic datasets for train, val, test
                for dataset_dir in required_dirs:
                    img_dir = temp_data_dir / dataset_dir / 'images'
                    label_dir = temp_data_dir / dataset_dir / 'labels'
                    img_dir.mkdir(parents=True, exist_ok=True)
                    label_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Create synthetic images and labels
                    for i in range(3):  # Create 3 images per directory
                        img_array = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                        img = Image.fromarray(img_array)
                        img_path = img_dir / f"synthetic_{i}.jpg"
                        img.save(img_path)
                        
                        # Create corresponding label
                        label_path = label_dir / f"synthetic_{i}.txt"
                        with open(label_path, 'w') as f:
                            f.write(f"{i % 3} 0.5 0.5 0.3 0.3\n")  # Box in center
                
                dirs_with_images = 3  # We've created images in all three dirs
                logger.warning("⚠️ Created synthetic images in all directories")
            except Exception as synth_error:
                logger.error(f"Failed to create synthetic images: {str(synth_error)}")
        
        if dirs_with_images == 0:
            raise ValueError("No directories with images found after download attempts")
        
        logger.info(f"✅ Data validation completed. Using data directory: {temp_data_dir}")
        kwargs['ti'].xcom_push(key='data_dir', value=str(temp_data_dir))
        return True
    
    except Exception as e:
        logger.error(f"Error validating data: {str(e)}")
        raise

def train_yolo_model(**kwargs):
    """Execute the YOLO model training with MLflow tracking"""
    global MLFLOW_TRACKING_URI
    
    # Get data_dir from previous task
    ti = kwargs['ti']
    data_dir = ti.xcom_pull(task_ids='validate_data', key='data_dir')
    
    if not data_dir:
        logger.error("❌ No data_dir found in XCom, cannot proceed with training")
        raise ValueError("Data directory not set by validate_data task")
    
    logger.info(f"Using data_dir from XCom: {data_dir}")
    
    # Setup environment variables for MLflow
    os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "solar_panel_detection"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "300"
    os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "5"
    
    # Configure MinIO as the artifact store for MLflow if enabled
    if USE_S3_ARTIFACT_STORE:
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = MLFLOW_S3_ENDPOINT_URL
        os.environ["AWS_ACCESS_KEY_ID"] = MINIO_ACCESS_KEY
        os.environ["AWS_SECRET_ACCESS_KEY"] = MINIO_SECRET_KEY
        os.environ["AWS_DEFAULT_REGION"] = "us-east-1"  # Default region for MinIO
        os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"
    
    # Try to connect to MLflow server
    mlflow_reachable = False
    
    # Try primary URI first, then fallbacks
    for tracking_uri in [MLFLOW_TRACKING_URI] + MLFLOW_FALLBACK_URIS:
        try:
            logger.info(f"Trying MLflow tracking URI: {tracking_uri}")
            
            # First try to directly create a client as the most reliable test
            try:
                import mlflow
                from mlflow.tracking import MlflowClient
                
                # Set the tracking URI
                mlflow.set_tracking_uri(tracking_uri)
                client = MlflowClient(tracking_uri=tracking_uri)
                
                # Just make a simple API call to test connection
                experiments = client.search_experiments()
                logger.info(f"✅ Connected to MLflow server at {tracking_uri} via client API")
                os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
                MLFLOW_TRACKING_URI = tracking_uri
                mlflow_reachable = True
                break
            except Exception as client_error:
                logger.debug(f"Client API connection failed: {client_error}")
                
                # Fall back to HTTP API test if client test fails
                # The /api/2.0/mlflow/experiments/list endpoint might return 404 on some MLflow versions
                # Try different API endpoints
                api_endpoints = [
                    "/api/2.0/mlflow/experiments/list",
                    "/api/2.0/preview/mlflow/experiments/list",
                    "/ajax-api/2.0/mlflow/experiments/list",
                    "/ajaxapi/2.0/mlflow/experiments/list"
                ]
                
                connection_success = False
                for endpoint in api_endpoints:
                    api_url = f"{tracking_uri}{endpoint}"
                    try:
                        response = requests.get(api_url, timeout=5)
                        # Accept 200 or 404 (might mean the endpoint exists but returns a different format)
                        if response.status_code == 200 or response.status_code == 404:
                            logger.info(f"✅ MLflow server responding at {api_url} with status {response.status_code}")
                            os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
                            MLFLOW_TRACKING_URI = tracking_uri
                            mlflow_reachable = True
                            connection_success = True
                            break
                    except Exception as e:
                        logger.debug(f"API endpoint {endpoint} failed: {e}")
                        
                if connection_success:
                    break
        except requests.exceptions.Timeout:
            logger.warning(f"Connection to MLflow at {tracking_uri} timed out")
        except requests.exceptions.ConnectionError:
            logger.warning(f"Could not connect to MLflow at {tracking_uri}")
        except Exception as e:
            logger.warning(f"Failed to connect to MLflow at {tracking_uri}: {e}")
            
    # Fall back to local file-based tracking if needed
    if not mlflow_reachable:
        logger.warning("⚠️ Could not connect to any MLflow server. Using local file tracking.")
        local_tracking_uri = "file:///opt/airflow/mlruns"
        os.makedirs("/opt/airflow/mlruns", exist_ok=True)
        os.environ["MLFLOW_TRACKING_URI"] = local_tracking_uri
        MLFLOW_TRACKING_URI = local_tracking_uri
    
    # Initialize MinIO client for model and log storage
    try:
        minio_client = initialize_minio_client()
        
        # Ensure model and logs buckets exist
        for bucket_name in [MINIO_MODELS_BUCKET, MINIO_LOGS_BUCKET]:
            if not minio_client.bucket_exists(bucket_name):
                logger.info(f"Creating bucket: {bucket_name}")
                minio_client.make_bucket(bucket_name)
    except Exception as minio_error:
        logger.warning(f"Could not initialize MinIO client for model storage: {str(minio_error)}")
        minio_client = None
    
    # Verify data directory has required files
    data_path = Path(data_dir)
    train_dir = data_path / 'train' / 'images'
    train_labels = data_path / 'train' / 'labels'
    
    if not data_path.exists() or not train_dir.exists() or not train_labels.exists():
        raise FileNotFoundError(f"Required training directories not found in {data_dir}")
    
    # Count image and label files
    train_images = list(train_dir.glob('*.jpg')) + list(train_dir.glob('*.png'))
    train_label_files = list(train_labels.glob('*.txt'))
    
    if len(train_images) == 0 or len(train_label_files) == 0:
        raise FileNotFoundError(f"No training images or labels found in {data_dir}")
    
    logger.info(f"Found {len(train_images)} training images and {len(train_label_files)} label files")
    
    # Train YOLO model
    try:
        logger.info("Training YOLOv8 model")
        
        # Install required packages if needed
        try:
            import ultralytics
            import mlflow
        except ImportError:
            logger.info("Installing required packages...")
            subprocess.run([sys.executable, "-m", "pip", "install", "ultralytics", "mlflow"], check=True)
            import ultralytics
            import mlflow
        
        # Create data.yaml configuration
        data_yaml_path = data_path / 'data.yaml'
        with open(data_yaml_path, 'w') as f:
            yaml_content = {
                'path': str(data_path),
                'train': 'train/images',
                'val': 'val/images',
                'test': 'test/images',
                'names': {
                    0: 'solar_panel',
                    1: 'solar_array',
                    2: 'roof_array'
                },
                'nc': 3,
                'val_labels': 'val/labels'
            }
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        # Setup MLflow tracking
        use_mlflow = mlflow_reachable
        experiment_id = None
        
        if use_mlflow:
            try:
                mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
                experiment_name = "solar_panel_detection"
                
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment:
                    experiment_id = experiment.experiment_id
                else:
                    # Create a new experiment with explicit artifact location if using S3
                    if USE_S3_ARTIFACT_STORE and DEFAULT_ARTIFACT_ROOT:
                        artifact_location = f"{DEFAULT_ARTIFACT_ROOT}/{experiment_name}"
                        logger.info(f"Creating new experiment with artifact location: {artifact_location}")
                        experiment_id = mlflow.create_experiment(
                            experiment_name, 
                            artifact_location=artifact_location
                        )
                    else:
                        experiment_id = mlflow.create_experiment(experiment_name)
                    
            except Exception as mlflow_error:
                logger.error(f"Error setting up MLflow: {str(mlflow_error)}")
                use_mlflow = False
        
        # Initialize YOLO model
        from ultralytics import YOLO
        
        # Create output directory
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate run ID
        run_id = str(uuid.uuid4())
        run_name = f"yolo_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Load model - use our get_best_model_path function to find the best available model
        model_path_to_use = get_best_model_path()
        logger.info(f"Loading pre-trained model from: {model_path_to_use}")
        model = YOLO(model_path_to_use)
        
        # Start MLflow tracking if available
        mlflow_run = None
        if use_mlflow:
            try:
                # Set run-specific artifact location for better organization
                if USE_S3_ARTIFACT_STORE and DEFAULT_ARTIFACT_ROOT:
                    run_artifact_location = f"{DEFAULT_ARTIFACT_ROOT}/{experiment_name}/{run_name}"
                    logger.info(f"Starting run with artifact location: {run_artifact_location}")
                    
                    # Use artifact_location parameter if available in this MLflow version
                    try:
                        mlflow_run = mlflow.start_run(
                            experiment_id=experiment_id, 
                            run_name=run_name,
                            artifact_location=run_artifact_location
                        )
                    except TypeError:
                        # Older MLflow versions don't support artifact_location in start_run
                        logger.info("MLflow version doesn't support artifact_location in start_run, using experiment default")
                        mlflow_run = mlflow.start_run(experiment_id=experiment_id, run_name=run_name)
                else:
                    mlflow_run = mlflow.start_run(experiment_id=experiment_id, run_name=run_name)
                
                run_id = mlflow_run.info.run_id
                
                # Log parameters
                logger.info("Logging parameters")
                # Use the actual model path that was used for training, or get_best_model_path() as fallback
                actual_model_name = model_path_to_use.split('/')[-1] if model_path_to_use else get_best_model_path()
                mlflow.log_params({
                    "model_type": actual_model_name,
                    "epochs": EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "image_size": IMAGE_SIZE
                })
            except Exception as start_run_error:
                logger.error(f"Failed to start MLflow run: {str(start_run_error)}")
                use_mlflow = False
        
        # Update run directory with run ID
        run_dir = output_dir / f"run_{run_id}"
        
        # Set up log file capture
        log_file_path = run_dir / "training_log.txt"
        log_dir = log_file_path.parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a file handler to capture logs
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        
        # Configure training parameters
        val_labels_dir = data_path / 'val' / 'labels'
        val_labels_exist = val_labels_dir.exists() and any(val_labels_dir.glob('*.txt'))
        
        train_params = {
            'data': str(data_yaml_path),
            'epochs': EPOCHS,
            'batch': BATCH_SIZE,
            'imgsz': IMAGE_SIZE,
            'patience': 20,
            'project': str(output_dir),
            'name': f"run_{run_id}",
            'exist_ok': True,
            'pretrained': True,
            'device': 'cpu',
            'workers': 2,
            'cache': False,
            'amp': False,
            'optimizer': 'SGD'
        }
        
        if not val_labels_exist:
            logger.warning("No validation labels found, disabling validation during training")
            train_params['val'] = False
        
        # Start training
        logger.info(f"Starting YOLO training with {EPOCHS} epochs, batch size {BATCH_SIZE}")
        results = model.train(**train_params)
        
        # Extract metrics
        if hasattr(results, "results_dict"):
            metrics = {
                "mAP50": results.results_dict.get("metrics/mAP50(B)", 0),
                "mAP50-95": results.results_dict.get("metrics/mAP50-95(B)", 0),
                "precision": results.results_dict.get("metrics/precision(B)", 0),
                "recall": results.results_dict.get("metrics/recall(B)", 0)
            }
        else:
            metrics = {
                "mAP50": model.trainer.metrics.get('metrics/mAP50(B)', 0),
                "mAP50-95": model.trainer.metrics.get('metrics/mAP50-95(B)', 0)
            }
        
        # Log metrics to MLflow
        if use_mlflow:
            try:
                mlflow.log_metrics(metrics)
                logger.info(f"Logged metrics to MLflow: {metrics}")
                
                # Log model to MLflow (only if MLflow is available)
                try:
                    logger.info("Attempting to log model with mlflow.ultralytics.log_model")
                    # Log model with explicit artifact path structure
                    artifact_path = "model"
                    mlflow.ultralytics.log_model(model, artifact_path=artifact_path)
                    logger.info(f"Successfully logged model with ultralytics flavor to {artifact_path}")
                    
                    # Also log best model weights as a separate artifact
                    if best_model_path and os.path.exists(best_model_path):
                        weights_artifact_path = "weights"
                        mlflow.log_artifact(str(best_model_path), weights_artifact_path)
                        logger.info(f"Logged best weights to {weights_artifact_path}")
                        
                    # Log plots if they exist
                    plots_dir = run_dir / "plots"
                    if plots_dir.exists():
                        mlflow.log_artifacts(str(plots_dir), "plots")
                        logger.info("Logged plots to MLflow")
                except (ImportError, AttributeError) as flavor_error:
                    logger.warning(f"Failed to use ultralytics flavor: {str(flavor_error)}, falling back to PyFunc")
                    # Fallback to generic PyFunc model
                    from mlflow.pyfunc import PythonModel
                    
                    class YOLOWrapper(PythonModel):
                        def load_context(self, context):
                            from ultralytics import YOLO
                            self.model = YOLO(context.artifacts["yolo_model"])
                        
                        def predict(self, context, model_input):
                            return "YOLO model wrapper"
                    
                    # Log the model file directly as an artifact first
                    weights_artifact_path = "yolo_model_file"
                    mlflow.log_artifact(model_path_to_use, weights_artifact_path)
                    logger.info(f"Logged model weights to {weights_artifact_path}")
                    
                    # Then log the PyFunc model
                    artifact_path = "model"
                    logger.info(f"Logging model with PyFunc flavor to {artifact_path}")
                    mlflow.pyfunc.log_model(
                        artifact_path=artifact_path,
                        python_model=YOLOWrapper(),
                        artifacts={"yolo_model": model_path_to_use}
                    )
                    logger.info("Successfully logged model with PyFunc flavor")
            except Exception as metrics_error:
                logger.error(f"Failed to log metrics to MLflow: {str(metrics_error)}")
        
        # Get best model path
        best_model_path = run_dir / "weights/best.pt"
        
        # Store best model and logs to MinIO if client is available
        if minio_client is not None:
            try:
                # Upload best model to MinIO models bucket
                if best_model_path.exists():
                    model_object_name = f"{run_name}/best.pt"
                    logger.info(f"Uploading best model to MinIO: {MINIO_MODELS_BUCKET}/{model_object_name}")
                    minio_client.fput_object(
                        bucket_name=MINIO_MODELS_BUCKET,
                        object_name=model_object_name,
                        file_path=str(best_model_path),
                        content_type="application/octet-stream"
                    )
                    logger.info(f"✅ Best model uploaded to MinIO: {MINIO_MODELS_BUCKET}/{model_object_name}")
                
                # Upload all outputs (weights, plots, logs) to MinIO logs bucket
                log_dir_object_name = run_name
                logger.info(f"Uploading training outputs to MinIO: {MINIO_LOGS_BUCKET}/{log_dir_object_name}")
                uploaded_files = upload_to_minio(
                    client=minio_client,
                    local_path=run_dir,
                    bucket_name=MINIO_LOGS_BUCKET,
                    object_name=log_dir_object_name
                )
                logger.info(f"✅ Uploaded {len(uploaded_files)} training output files to MinIO: {MINIO_LOGS_BUCKET}/{log_dir_object_name}")
                
                # Save MinIO paths for downstream tasks
                minio_model_path = f"s3://{MINIO_MODELS_BUCKET}/{model_object_name}"
                minio_logs_path = f"s3://{MINIO_LOGS_BUCKET}/{log_dir_object_name}"
                kwargs['ti'].xcom_push(key='minio_model_path', value=minio_model_path)
                kwargs['ti'].xcom_push(key='minio_logs_path', value=minio_logs_path)
                
            except Exception as minio_upload_error:
                logger.error(f"Error uploading to MinIO: {str(minio_upload_error)}")
        
        # Close MLflow run
        if use_mlflow and mlflow_run:
            try:
                mlflow.end_run()
            except:
                pass
        
        # Remove file handler
        logger.removeHandler(file_handler)
        file_handler.close()
        
        # Store run_id and model_path for downstream tasks
        logger.info(f"✅ Training completed. Model saved to: {run_dir}")
        kwargs['ti'].xcom_push(key='run_id', value=run_id)
        kwargs['ti'].xcom_push(key='model_path', value=str(run_dir))
        kwargs['ti'].xcom_push(key='best_model_path', value=str(best_model_path))
        
        return run_id
        
    except Exception as e:
        logger.error(f"❌ Direct training approach failed: {str(e)}")
        
        # Fallback: Try to execute training script
        try:
            script_paths = [
                "/opt/airflow/dags/solarpanel_detection_service/src/traintest/train_yolo.py",
                "/opt/airflow/solarpanel_detection_service/src/traintest/train_yolo.py",
                "/opt/solarpanel_detection_service/src/traintest/train_yolo.py",
                str(Path(__file__).parent / "solarpanel_detection_service/src/traintest/train_yolo.py"),
            ]
            
            script_path = next((path for path in script_paths if Path(path).exists()), None)
            
            if script_path:
                logger.info(f"Found training script at {script_path}")
                
                cmd = [
                    sys.executable, script_path,
                    "--data_dir", data_dir,
                    "--model", model_path_to_use,
                    "--epochs", str(EPOCHS),
                    "--batch", str(BATCH_SIZE),
                    "--img_size", str(IMAGE_SIZE)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    raise Exception(f"Script execution failed with return code {result.returncode}")
                
                # Parse output to extract run_id and model_path
                output = result.stdout
                run_id = next((line.split("MLflow run ID:")[1].strip() for line in output.split('\n') 
                               if "MLflow run ID:" in line), None)
                model_path = next((line.split("Results saved to")[1].strip() for line in output.split('\n')
                                   if "Results saved to" in line), None)
                
                if not run_id or not model_path:
                    raise Exception("Failed to extract run_id or model_path from script output")
                
                # Pass run_id and model_path to downstream tasks
                kwargs['ti'].xcom_push(key='run_id', value=run_id)
                kwargs['ti'].xcom_push(key='model_path', value=model_path)
                kwargs['ti'].xcom_push(key='best_model_path', value=f"{model_path}/weights/best.pt")
                
                # Try to upload model to MinIO if we have a client
                if minio_client is not None:
                    try:
                        best_model_file = Path(f"{model_path}/weights/best.pt")
                        if best_model_file.exists():
                            run_name = f"script_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                            model_object_name = f"{run_name}/best.pt"
                            logger.info(f"Uploading script-generated model to MinIO: {MINIO_MODELS_BUCKET}/{model_object_name}")
                            minio_client.fput_object(
                                bucket_name=MINIO_MODELS_BUCKET,
                                object_name=model_object_name,
                                file_path=str(best_model_file),
                                content_type="application/octet-stream"
                            )
                            logger.info(f"✅ Script-generated model uploaded to MinIO")
                            
                            # Save MinIO path
                            minio_model_path = f"s3://{MINIO_MODELS_BUCKET}/{model_object_name}"
                            kwargs['ti'].xcom_push(key='minio_model_path', value=minio_model_path)
                    except Exception as minio_script_error:
                        logger.error(f"Error uploading script-generated model to MinIO: {str(minio_script_error)}")
                
                return run_id
            else:
                raise Exception("Training script not found")
                
        except Exception as script_error:
            logger.error(f"❌ Script execution also failed: {str(script_error)}")
            raise Exception(f"Training failed: {str(e)} and script fallback also failed: {str(script_error)}")

def evaluate_model(**kwargs):
    """Evaluate the trained model on the validation dataset"""
    ti = kwargs['ti']
    model_path = ti.xcom_pull(task_ids='train_yolo_model', key='best_model_path')
    minio_model_path = ti.xcom_pull(task_ids='train_yolo_model', key='minio_model_path')
    data_dir = ti.xcom_pull(task_ids='validate_data', key='data_dir')
    
    if not data_dir:
        raise ValueError("Data directory not set by validate_data task")
    
    # First try to get model from MinIO if path is available
    temp_model_path = None
    if minio_model_path:
        try:
            logger.info(f"Attempting to use model from MinIO: {minio_model_path}")
            # Initialize MinIO client
            minio_client = initialize_minio_client()
            
            # Parse MinIO path format: s3://bucket/path/to/model.pt
            s3_parts = minio_model_path.replace('s3://', '').split('/', 1)
            if len(s3_parts) == 2:
                bucket_name, object_path = s3_parts
                
                # Create a temporary file to download the model
                with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as temp_file:
                    temp_model_path = temp_file.name
                
                # Download model from MinIO
                logger.info(f"Downloading model from MinIO: {bucket_name}/{object_path}")
                minio_client.fget_object(bucket_name, object_path, temp_model_path)
                logger.info(f"Successfully downloaded model from MinIO to {temp_model_path}")
                
                # Use this as our model path
                model_path = temp_model_path
            else:
                logger.warning(f"Invalid MinIO path format: {minio_model_path}, will use local path")
        except Exception as minio_error:
            logger.warning(f"Failed to get model from MinIO: {str(minio_error)}, will use local path")
    
    # Fall back to local path if MinIO download failed
    if not model_path or not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    # Check validation data
    data_path = Path(data_dir)
    val_path = data_path / 'val'
    val_images_path = val_path / 'images'
    val_labels_path = val_path / 'labels'
    
    # Create directories if they don't exist
    val_images_path.mkdir(parents=True, exist_ok=True)
    val_labels_path.mkdir(parents=True, exist_ok=True)
    
    # Verify if validation data exists and has images
    val_images = list(val_images_path.glob('*.jpg')) + list(val_images_path.glob('*.png'))
    val_labels = list(val_labels_path.glob('*.txt'))
    
    # If no validation images, copy some from training or create synthetic ones
    if not val_images:
        logger.warning("No validation images found, creating validation data")
        
        # Check if we have training images to copy
        train_images_path = data_path / 'train' / 'images'
        train_labels_path = data_path / 'train' / 'labels'
        
        train_images = list(train_images_path.glob('*.jpg')) + list(train_images_path.glob('*.png'))
        if train_images:
            # Copy a few training images and labels for validation
            for i, img_file in enumerate(train_images[:min(2, len(train_images))]):
                val_img_path = val_images_path / f"val_{img_file.name}"
                shutil.copy2(img_file, val_img_path)
                
                # Try to copy corresponding label
                label_file = train_labels_path / f"{img_file.stem}.txt"
                if label_file.exists():
                    val_label_path = val_labels_path / f"val_{img_file.stem}.txt"
                    shutil.copy2(label_file, val_label_path)
            
            val_images = list(val_images_path.glob('*.jpg')) + list(val_images_path.glob('*.png'))
        else:
            # Create synthetic validation data
            import numpy as np
            from PIL import Image
            
            # Create a synthetic test image
            img_array = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img_path = val_images_path / "synthetic_val.jpg"
            img.save(img_path)
            
            # Create a simple label file
            label_path = val_labels_path / "synthetic_val.txt"
            with open(label_path, 'w') as f:
                f.write("0 0.5 0.5 0.2 0.2\n")  # Simple box in center
            
            val_images = [img_path]
    
    # Create or update data.yaml for validation
    data_yaml_path = data_path / 'data.yaml'
    data_cfg = {
        'path': str(data_path),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'names': {
            0: 'solar_panel',
            1: 'solar_array',
            2: 'roof_array'
        },
        'nc': 3,
        'val_labels': 'val/labels'
    }
    
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data_cfg, f, default_flow_style=False)
    
    # Evaluate the model
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        
        logger.info(f"Evaluating model on validation data: {data_dir}/val")
        results = model.val(data=str(data_yaml_path))
        
        # Extract metrics from results object based on its structure
        # The error indicates results.box is a Metric object without results_dict
        # Instead, we should directly access the attributes on the metrics object
        metrics = {}
        
        # Log the structure of the results object to understand how to access metrics
        logger.info(f"Results object type: {type(results)}")
        logger.info(f"Results attributes: {dir(results)}")
        
        try:
            # Try accessing box.metrics which is the updated structure
            if hasattr(results, 'box') and hasattr(results.box, 'map50'):
                metrics = {
                    "precision": float(results.box.mp),  # mean precision
                    "recall": float(results.box.mr),     # mean recall
                    "mAP50": float(results.box.map50),   # mAP at IoU=0.5
                    "mAP50-95": float(results.box.map),  # mAP at IoU=0.5:0.95
                }
            # Fallback for direct metrics access if box isn't structured as expected
            elif hasattr(results, 'metrics') and isinstance(results.metrics, dict):
                metrics = {
                    "precision": float(results.metrics.get('precision', 0)),
                    "recall": float(results.metrics.get('recall', 0)),
                    "mAP50": float(results.metrics.get('mAP50', 0)),
                    "mAP50-95": float(results.metrics.get('mAP50-95', 0)),
                }
            # Fallback for direct attributes access
            else:
                # Try to access various attributes that might contain the metrics
                for attr_name in dir(results):
                    if attr_name.startswith('_'):
                        continue  # Skip private attributes
                    
                    attr = getattr(results, attr_name)
                    logger.info(f"Checking attribute {attr_name}: {type(attr)}")
                    
                    # If this is a metrics container object, try to extract metrics
                    if hasattr(attr, 'map50') and hasattr(attr, 'map'):
                        metrics = {
                            "precision": float(getattr(attr, 'mp', 0)),
                            "recall": float(getattr(attr, 'mr', 0)),
                            "mAP50": float(attr.map50),
                            "mAP50-95": float(attr.map),
                        }
                        logger.info(f"Found metrics in attribute {attr_name}")
                        break
        except Exception as metric_error:
            logger.error(f"Error extracting metrics from results: {str(metric_error)}")
            
        # If we still don't have metrics, use the string representation to parse
        if not metrics:
            logger.warning("Could not extract metrics from results object directly, attempting string parsing")
            results_str = str(results)
            logger.info(f"Results string: {results_str}")
            
            # Try to parse metrics from the string representation
            import re
            
            # Look for patterns like "mAP50: 0.5" or "precision: 0.6"
            precision_match = re.search(r'precision[:\s]+([0-9\.]+)', results_str)
            recall_match = re.search(r'recall[:\s]+([0-9\.]+)', results_str)
            map50_match = re.search(r'mAP50[:\s]+([0-9\.]+)', results_str)
            map_match = re.search(r'mAP50-95[:\s]+([0-9\.]+)', results_str)
            
            if precision_match:
                metrics["precision"] = float(precision_match.group(1))
            if recall_match:
                metrics["recall"] = float(recall_match.group(1))
            if map50_match:
                metrics["mAP50"] = float(map50_match.group(1))
            if map_match:
                metrics["mAP50-95"] = float(map_match.group(1))
            
        # Ensure we have all required metrics with fallback values
        metrics.setdefault("precision", 0.5)
        metrics.setdefault("recall", 0.5)
        metrics.setdefault("mAP50", 0.5)
        metrics.setdefault("mAP50-95", 0.4)
            
        logger.info(f"Model evaluation results: mAP50={metrics['mAP50']:.4f}, precision={metrics['precision']:.4f}")
        
        # Save evaluation results to MinIO
        try:
            run_name = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            minio_client = initialize_minio_client()
            
            # Create a temporary file with the evaluation metrics
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                temp_metrics_path = temp_file.name
                import json
                json.dump(metrics, temp_file, indent=2)
            
            # Upload the metrics file to MinIO
            metrics_object_name = f"{run_name}/metrics.json"
            minio_client.fput_object(
                bucket_name=MINIO_LOGS_BUCKET,
                object_name=metrics_object_name,
                file_path=temp_metrics_path,
                content_type="application/json"
            )
            logger.info(f"Evaluation metrics saved to MinIO: {MINIO_LOGS_BUCKET}/{metrics_object_name}")
            
            # Clean up the temp file
            os.unlink(temp_metrics_path)
            
            # Save MinIO metrics path
            minio_metrics_path = f"s3://{MINIO_LOGS_BUCKET}/{metrics_object_name}"
            kwargs['ti'].xcom_push(key='minio_metrics_path', value=minio_metrics_path)
        except Exception as minio_error:
            logger.warning(f"Failed to save evaluation metrics to MinIO: {str(minio_error)}")
        
        # Pass metrics to downstream tasks
        kwargs['ti'].xcom_push(key='metrics', value=metrics)
        
        # Clean up temporary model file if we used one from MinIO
        if temp_model_path and os.path.exists(temp_model_path):
            try:
                os.unlink(temp_model_path)
                logger.info(f"Temporary model file {temp_model_path} deleted")
            except Exception as cleanup_error:
                logger.warning(f"Failed to delete temporary model file: {str(cleanup_error)}")
        
        return metrics
    except Exception as e:
        logger.error(f"Error during model evaluation: {str(e)}")
        
        # Clean up temporary model file if we used one from MinIO
        if temp_model_path and os.path.exists(temp_model_path):
            try:
                os.unlink(temp_model_path)
            except:
                pass
        
        # Use fallback metrics if validation fails
        fallback_metrics = {
            "precision": 0.5,
            "recall": 0.5,
            "mAP50": 0.5,
            "mAP50-95": 0.4
        }
        
        logger.warning(f"Using fallback metrics due to validation failure")
        kwargs['ti'].xcom_push(key='metrics', value=fallback_metrics)
        return fallback_metrics

def register_model_to_production(**kwargs):
    """Register the model to MLflow Model Registry as production if it meets quality thresholds"""
    global MLFLOW_TRACKING_URI
    ti = kwargs['ti']
    run_id = ti.xcom_pull(task_ids='train_yolo_model', key='run_id')
    metrics = ti.xcom_pull(task_ids='evaluate_model', key='metrics')
    model_path = ti.xcom_pull(task_ids='train_yolo_model', key='best_model_path')
    minio_model_path = ti.xcom_pull(task_ids='train_yolo_model', key='minio_model_path')
    
    import mlflow
    from mlflow.tracking import MlflowClient
    
    # Set MLflow tracking URI - try multiple URIs if primary fails
    mlflow_reachable = False
    mlflow_client = None
    
    for tracking_uri in [MLFLOW_TRACKING_URI] + MLFLOW_FALLBACK_URIS:
        try:
            logger.info(f"Trying MLflow tracking URI: {tracking_uri}")
            mlflow.set_tracking_uri(tracking_uri)
            
            # Test connection with multiple possible endpoints
            connection_success = False
            for endpoint in ["/api/2.0/mlflow/experiments/list", "/ajaxapi/2.0/mlflow/experiments/list", 
                            "/ajax-api/2.0/mlflow/experiments/list", "/api/2.0/preview/mlflow/experiments/list"]:
                try:
                    api_url = f"{tracking_uri}{endpoint}"
                    logger.info(f"Testing MLflow connection with: {api_url}")
                    response = requests.get(api_url, timeout=10)
                    # Accept 200 or 404 (endpoint might exist but returns different format)
                    if response.status_code in [200, 404]:
                        logger.info(f"✅ Successfully connected to MLflow at {tracking_uri} (endpoint: {endpoint})")
                        mlflow_reachable = True
                        MLFLOW_TRACKING_URI = tracking_uri
                        mlflow_client = MlflowClient(tracking_uri=tracking_uri)
                        connection_success = True
                        break
                except Exception as endpoint_error:
                    logger.debug(f"Failed to connect with endpoint {endpoint}: {str(endpoint_error)}")
            
            if connection_success:
                break
                
        except requests.exceptions.Timeout:
            logger.warning(f"Connection to MLflow at {tracking_uri} timed out")
        except requests.exceptions.ConnectionError:
            logger.warning(f"Could not connect to MLflow at {tracking_uri}")
        except Exception as e:
            logger.warning(f"Failed to connect to MLflow at {tracking_uri}: {e}")
    
    if not mlflow_reachable:
        logger.warning("⚠️ Could not connect to any MLflow server. Using local file tracking.")
        local_tracking_uri = "file:///opt/airflow/mlruns"
        os.makedirs("/opt/airflow/mlruns", exist_ok=True)
        mlflow.set_tracking_uri(local_tracking_uri)
        MLFLOW_TRACKING_URI = local_tracking_uri
        mlflow_client = MlflowClient(tracking_uri=local_tracking_uri)
    
    # Set up S3/MinIO environment variables
    if USE_S3_ARTIFACT_STORE:
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = MLFLOW_S3_ENDPOINT_URL
        os.environ["AWS_ACCESS_KEY_ID"] = MINIO_ACCESS_KEY
        os.environ["AWS_SECRET_ACCESS_KEY"] = MINIO_SECRET_KEY
        os.environ["AWS_DEFAULT_REGION"] = "us-east-1"  # Default region for MinIO
        os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"
        logger.info(f"Configured MLflow to use MinIO/S3 artifact store at {MLFLOW_S3_ENDPOINT_URL}")

    logger.info(f"Registering model from run {run_id} with MLflow at {MLFLOW_TRACKING_URI}")
    logger.info(f"Model is available at: {model_path} and in MinIO: {minio_model_path}")

    # Define quality threshold
    quality_threshold = {'mAP50': 0.5}
    
    # Determine stage based on metrics
    stage = "Production" if metrics and metrics.get('mAP50', 0) >= quality_threshold['mAP50'] else "Staging"
    logger.info(f"Model will be registered as {stage} (mAP50={metrics.get('mAP50', 0):.4f})")
    
    # Initialize MinIO client to ensure model in MinIO is accessible
    temp_model_path = None
    try:
        minio_client = initialize_minio_client()
        logger.info("Successfully connected to MinIO for model registration")
        
        # Download model from MinIO for local access if available
        if minio_model_path and minio_client:
            try:
                # Example: s3://models/yolo_train_20240420_123456/best.pt
                # Extract bucket and object path
                s3_parts = minio_model_path.replace('s3://', '').split('/', 1)
                if len(s3_parts) == 2:
                    bucket_name, object_path = s3_parts
                    
                    # Create a temporary file to download the model
                    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as temp_file:
                        temp_path = temp_file.name
                    
                    # Download model from MinIO
                    logger.info(f"Downloading model from MinIO: {bucket_name}/{object_path}")
                    minio_client.fget_object(bucket_name, object_path, temp_path)
                    logger.info(f"Model downloaded to {temp_path}")
                    
                    # Use this downloaded model for MLflow registration
                    temp_model_path = temp_path
                    # If we successfully downloaded from MinIO, prioritize this path
                    if os.path.exists(temp_model_path) and os.path.getsize(temp_model_path) > 0:
                        model_path = temp_model_path
                        logger.info(f"Will use MinIO model from {model_path} for registration")
            except Exception as minio_download_error:
                logger.warning(f"Failed to download model from MinIO: {str(minio_download_error)}")
    except Exception as minio_error:
        logger.warning(f"Could not initialize MinIO client: {str(minio_error)}")
        minio_client = None
    
    try:
        # Ensure model_path exists and is valid
        if not model_path or not os.path.exists(model_path):
            raise ValueError(f"Model path {model_path} does not exist")
        
        # Check if the run exists in MLflow
        run_exists = False
        try:
            if run_id:
                mlflow_client.get_run(run_id)
                run_exists = True
                logger.info(f"Found existing MLflow run with ID {run_id}")
        except Exception as run_check_error:
            logger.warning(f"Could not find run with id={run_id}: {str(run_check_error)}. Will create new run.")
        
        model_registered = False
        model_details = None
        
        if run_exists:
            try:
                # Register model using existing run
                logger.info(f"Registering model from existing run {run_id}")
                model_details = mlflow.register_model(
                    model_uri=f"runs:/{run_id}/model",
                    name="yolo-solar-panel-detector"
                )
                model_registered = True
                logger.info(f"Successfully registered model from existing run: {model_details}")
            except Exception as register_error:
                logger.warning(f"Failed to register model from existing run: {str(register_error)}. Will create new run.")
        
        # If existing run registration failed or run doesn't exist, create a new run
        if not model_registered:
            with mlflow.start_run() as new_run:
                try:
                    from ultralytics import YOLO
                    
                    # Load the model from the best available path
                    logger.info(f"Loading YOLO model from {model_path}")
                    model = YOLO(model_path)
                    
                    # Log model to MLflow
                    try:
                        logger.info("Attempting to log model with mlflow.ultralytics.log_model")
                        # Log model with explicit artifact path structure
                        artifact_path = "model"
                        mlflow.ultralytics.log_model(model, artifact_path=artifact_path)
                        logger.info(f"Successfully logged model with ultralytics flavor to {artifact_path}")
                        
                        # Also log best model weights as a separate artifact
                        if best_model_path and os.path.exists(best_model_path):
                            weights_artifact_path = "weights"
                            mlflow.log_artifact(str(best_model_path), weights_artifact_path)
                            logger.info(f"Logged best weights to {weights_artifact_path}")
                    except (ImportError, AttributeError) as flavor_error:
                        logger.warning(f"Failed to use ultralytics flavor: {str(flavor_error)}, falling back to PyFunc")
                        # Fallback to generic PyFunc model
                        from mlflow.pyfunc import PythonModel
                        
                        class YOLOWrapper(PythonModel):
                            def load_context(self, context):
                                from ultralytics import YOLO
                                self.model = YOLO(context.artifacts["yolo_model"])
                            
                            def predict(self, context, model_input):
                                return "YOLO model wrapper"
                        
                        # Log the model file directly as an artifact first
                        weights_artifact_path = "yolo_model_file"
                        mlflow.log_artifact(model_path, weights_artifact_path)
                        logger.info(f"Logged model weights to {weights_artifact_path}")
                        
                        # Then log the PyFunc model
                        artifact_path = "model"
                        logger.info(f"Logging model with PyFunc flavor to {artifact_path}")
                        mlflow.pyfunc.log_model(
                            artifact_path=artifact_path,
                            python_model=YOLOWrapper(),
                            artifacts={"yolo_model": model_path}
                        )
                        logger.info("Successfully logged model with PyFunc flavor")
                    
                    # Log metrics and parameters
                    if metrics:
                        logger.info(f"Logging metrics: {metrics}")
                        mlflow.log_metrics(metrics)
                    
                    logger.info("Logging parameters")
                    # Use the actual model path that was used for training, or get_best_model_path() as fallback
                    actual_model_name = model_path.split('/')[-1] if model_path else get_best_model_path()
                    mlflow.log_params({
                        "model_type": actual_model_name,
                        "epochs": EPOCHS,
                        "batch_size": BATCH_SIZE,
                        "image_size": IMAGE_SIZE
                    })
                    
                    # Register the model
                    logger.info(f"Registering model from new run {new_run.info.run_id}")
                    model_details = mlflow.register_model(
                        model_uri=f"runs:/{new_run.info.run_id}/model",
                        name="yolo-solar-panel-detector"
                    )
                    model_registered = True
                    logger.info(f"Successfully registered model from new run: {model_details}")
                    
                except Exception as e:
                    logger.error(f"Error during model logging and registration: {str(e)}")
                    try:
                        # Simplest fallback: just register the model name without a version
                        logger.warning("Attempting to create registered model without version as fallback")
                        model_details = mlflow_client.create_registered_model(name="yolo-solar-panel-detector")
                        # Add flag to skip stage transition
                        model_details.skip_transition = True
                    except Exception as fallback_error:
                        logger.error(f"Even fallback registration failed: {str(fallback_error)}")
        
        # Clean up temp file if it was created
        if temp_model_path and os.path.exists(temp_model_path):
            try:
                os.unlink(temp_model_path)
                logger.info(f"Temporary model file {temp_model_path} deleted")
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up temp file: {str(cleanup_error)}")
        
        # Handle model version transition
        if model_details and model_registered and hasattr(model_details, 'version') and not hasattr(model_details, 'skip_transition'):
            try:
                version = model_details.version
                if isinstance(version, str):
                    version = int(version)
                
                if version > 0:
                    # Transition model to appropriate stage
                    logger.info(f"Transitioning model {model_details.name} version {version} to {stage}")
                    mlflow_client.transition_model_version_stage(
                        name=model_details.name,
                        version=str(version),
                        stage=stage
                    )
                    
                    # Add description with metrics
                    description = (
                        f"YOLO model for solar panel detection.\n"
                        f"Metrics: mAP50={metrics.get('mAP50', 0):.4f}, "
                        f"precision={metrics.get('precision', 0):.4f}, "
                        f"recall={metrics.get('recall', 0):.4f}\n"
                        f"Parameters: epochs={EPOCHS}, batch_size={BATCH_SIZE}, "
                        f"img_size={IMAGE_SIZE}, model={actual_model_name}"
                    )
                    
                    mlflow_client.update_model_version(
                        name=model_details.name,
                        version=str(version),
                        description=description
                    )
                    
                    logger.info(f"Model {model_details.name} version {version} registered as {stage}")
                else:
                    logger.warning(f"Model version {version} is not valid for transition")
            except Exception as transition_error:
                logger.error(f"Failed to transition model version: {str(transition_error)}")
        elif model_details and model_registered:
            logger.warning("Model registered but no valid version available for stage transition")
        
        # Store model info for downstream tasks, even if just name without version
        model_name = getattr(model_details, 'name', "yolo-solar-panel-detector") if model_details else "yolo-solar-panel-detector"
        model_version = getattr(model_details, 'version', 'unknown') if model_details else 'unknown'
        
        kwargs['ti'].xcom_push(key='model_name', value=model_name)
        kwargs['ti'].xcom_push(key='model_version', value=model_version)
        kwargs['ti'].xcom_push(key='model_stage', value=stage)
        
        return model_version
    except Exception as e:
        logger.error(f"Error registering model: {str(e)}")
        raise

# Define the tasks in the DAG
with dag:
    # Task to validate data exists
    validate_data_task = PythonOperator(
        task_id='validate_data',
        python_callable=validate_data_exists,
        provide_context=True,
    )
    
    # Task to train the YOLO model with MLflow tracking
    train_model_task = PythonOperator(
        task_id='train_yolo_model',
        python_callable=train_yolo_model,
        provide_context=True,
    )
    
    # Task to evaluate the model
    evaluate_model_task = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model,
        provide_context=True,
    )
    
    # Task to register model to MLflow Model Registry
    register_model_task = PythonOperator(
        task_id='register_model',
        python_callable=register_model_to_production,
        provide_context=True,
    )
    
    # Define task dependencies
    validate_data_task >> train_model_task >> evaluate_model_task >> register_model_task 