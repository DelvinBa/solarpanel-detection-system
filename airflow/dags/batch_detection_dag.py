import os
import cv2
import logging
import pandas as pd
import io
import tempfile
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from minio import Minio
from ultralytics import YOLO
import requests
from airflow.models import Variable
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment configuration
MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'minio:9000')
MINIO_PORT = os.getenv('MINIO_PORT', '9000')
MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY', 'minioadmin')
BUCKET_NAME = os.getenv('MINIO_BUCKET', 'inference-data')
MODELS_BUCKET = os.getenv('MODELS_BUCKET', 'models')
MINIO_SECURE = os.getenv('MINIO_SECURE', 'False').lower() == 'true'
INFERENCE_IMAGES_FOLDER = "inference_images/"
DETECTION_RESULTS_FOLDER = "detection_results/"
MANIFEST_FILENAME = "house_id_results.csv"

# Path to YOLO model (as fallback)
DAGS_FOLDER = os.path.dirname(os.path.abspath(__file__))
YOLO_MODEL_PATH = os.path.join(DAGS_FOLDER, "models", "best5.pt")

# MLflow configuration - need to get from train_yolo_mlflow_dag.py
def get_mlflow_tracking_uri():
    """Determine the appropriate MLflow tracking URI based on environment"""
    # First check if explicitly set in Airflow variables
    uri_from_variable = Variable.get('mlflow_tracking_uri', default_var=None)
    if uri_from_variable and uri_from_variable != "http://3.88.102.215:5001":
        return uri_from_variable
        
    # Check if we're on EC2 (simple check)
    try:
        response = requests.get('http://169.254.169.254/latest/meta-data/instance-id', timeout=0.1)
        if response.status_code == 200:
            # Use the EC2 MLflow server IP from a dedicated Variable
            ec2_mlflow_ip = Variable.get('mlflow_ec2_ip', default_var="172.31.21.44")
            ec2_mlflow_port = Variable.get('mlflow_ec2_port', default_var="5000")
            ec2_mlflow_uri = f"http://{ec2_mlflow_ip}:{ec2_mlflow_port}"
            logger.info(f"Running on EC2, using MLflow server at {ec2_mlflow_uri}")
            return ec2_mlflow_uri
    except:
        pass
    
    # For local/development environment, prefer localhost or docker service name
    local_uri = Variable.get('mlflow_local_uri', default_var="http://tracking_server:5000")
    logger.info(f"Running in local/dev environment, using MLflow server at {local_uri}")
    return local_uri

# Add fallback MLflow tracking URI options
MLFLOW_TRACKING_URI = get_mlflow_tracking_uri()
MLFLOW_FALLBACK_URIS = [
    Variable.get('mlflow_local_uri', default_var="http://tracking_server:5000"),
    "http://localhost:5000",
    "http://127.0.0.1:5000",
    "http://tracking_server:5001",
    "file:///opt/airflow/mlruns"  # Local file-based tracking as last resort
]

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 3, 26),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(minutes=30),
    'max_active_runs': 1,
    'max_active_tasks': 1,
}

def initialize_minio_client():
    """Initialize and return a MinIO client."""
    try:
        endpoint = MINIO_ENDPOINT
        # Always include the port in the endpoint, regardless of hostname
        endpoint = f"{MINIO_ENDPOINT}:{MINIO_PORT}"
        
        logger.info(f"Attempting to connect to MinIO/S3 at {endpoint} (secure={MINIO_SECURE})")
        
        # Initialize MinIO client
        client = Minio(
            endpoint=endpoint,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE
        )
        logger.info("MinIO client initialized successfully.")
        
        # Ensure bucket exists
        try:
            if not client.bucket_exists(BUCKET_NAME):
                logger.info(f"Bucket '{BUCKET_NAME}' does not exist. Attempting to create it.")
                client.make_bucket(BUCKET_NAME)
                logger.info(f"Bucket '{BUCKET_NAME}' created successfully.")
            else:
                logger.info(f"Bucket '{BUCKET_NAME}' already exists.")
        except Exception as bucket_error:
            logger.error(f"Error with bucket '{BUCKET_NAME}': {str(bucket_error)}")
            # Try creating required folders in existing buckets
            try:
                # Create some test content to ensure we have write access
                logger.info(f"Attempting to create test folders in '{BUCKET_NAME}'")
                client.put_object(BUCKET_NAME, f"{INFERENCE_IMAGES_FOLDER}.test", io.BytesIO(b""), 0)
                client.put_object(BUCKET_NAME, f"{DETECTION_RESULTS_FOLDER}.test", io.BytesIO(b""), 0)
                logger.info("Successfully created test folders. We have write access.")
            except Exception as folder_error:
                logger.error(f"Error creating test folders: {str(folder_error)}")
                raise
                
        return client
    except Exception as e:
        logger.error(f"Error initializing MinIO client: {e}")
        raise

def get_latest_model(client):
    """
    Find the latest YOLO model in the following order:
    1. Try to get the latest Production model from MLflow registry
    2. Fall back to MinIO models bucket if MLflow fails
    3. Use the fallback hardcoded model path as last resort
    
    Returns the model object name and a local path to the downloaded model.
    """
    try:
        # First try MLflow - import here to avoid issues if not installed
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
            
            mlflow_reachable = False
            
            # Try different MLflow tracking URIs
            for tracking_uri in [MLFLOW_TRACKING_URI] + MLFLOW_FALLBACK_URIS:
                try:
                    logger.info(f"Trying MLflow tracking URI: {tracking_uri}")
                    mlflow.set_tracking_uri(tracking_uri)
                    
                    # First try to directly create a client as the most reliable test
                    try:
                        mlflow_client = MlflowClient(tracking_uri=tracking_uri)
                        # Just make a simple API call to test connection
                        experiments = mlflow_client.search_experiments()
                        logger.info(f"Connected to MLflow server at {tracking_uri} via client API")
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
                        
                        for endpoint in api_endpoints:
                            api_url = f"{tracking_uri}{endpoint}"
                            try:
                                response = requests.get(api_url, timeout=5)
                                # Accept 200 or 404 (might mean the endpoint exists but returns a different format)
                                if response.status_code == 200 or response.status_code == 404:
                                    logger.info(f"MLflow server responding at {api_url} with status {response.status_code}")
                                    mlflow_reachable = True
                                    break
                            except Exception as e:
                                logger.debug(f"API endpoint {endpoint} failed: {e}")
                                
                        if mlflow_reachable:
                            break
                except requests.exceptions.Timeout:
                    logger.warning(f"Connection to MLflow at {tracking_uri} timed out")
                except requests.exceptions.ConnectionError:
                    logger.warning(f"Could not connect to MLflow at {tracking_uri}")
                except Exception as e:
                    logger.warning(f"Failed to connect to MLflow at {tracking_uri}: {e}")
            
            if mlflow_reachable:
                # Try to get the latest Production model
                mlflow_client = MlflowClient()
                model_name = "yolo-solar-panel-detector"  # Match name in training DAG
                
                # Find models in Production stage first
                try:
                    production_versions = mlflow_client.get_latest_versions(model_name, stages=["Production"])
                    if production_versions:
                        model_version = production_versions[0].version
                        logger.info(f"Found Production model {model_name} version {model_version}")
                        
                        # Download the model
                        _, temp_file_path = tempfile.mkstemp(suffix='.pt')
                        model_uri = f"models:/{model_name}/{model_version}"
                        
                        # This downloads the model to the specified path
                        model_path = mlflow.artifacts.download_artifacts(
                            artifact_uri=model_uri,
                            dst_path=os.path.dirname(temp_file_path)
                        )
                        
                        # Navigate to the actual model file (need to find the .pt file)
                        if os.path.isdir(model_path):
                            # Look for .pt files in the directory
                            model_files = []
                            for root, _, files in os.walk(model_path):
                                for file in files:
                                    if file.endswith('.pt'):
                                        model_files.append(os.path.join(root, file))
                            
                            if model_files:
                                logger.info(f"Found MLflow model file: {model_files[0]}")
                                return f"mlflow:{model_name}/version/{model_version}", model_files[0]
                        
                        logger.warning("Downloaded MLflow model but couldn't find .pt file")
                except Exception as mlflow_error:
                    logger.warning(f"Error getting model from MLflow: {mlflow_error}")
        except ImportError:
            logger.warning("MLflow not available, skipping MLflow model check")
        except Exception as e:
            logger.warning(f"General error with MLflow: {e}")
        
        # Fall back to MinIO if MLflow didn't work
        logger.info("Falling back to MinIO for model retrieval")
        
        # Make sure we're using the minio client, not the mlflow_client
        minio_client = client  # Use the passed-in MinIO client
        
        # Check if models bucket exists
        if not minio_client.bucket_exists(MODELS_BUCKET):
            logger.warning(f"Models bucket '{MODELS_BUCKET}' does not exist. Using fallback model.")
            return None, YOLO_MODEL_PATH
        
        # List all objects in the models bucket
        logger.info(f"Listing objects in '{MODELS_BUCKET}' bucket to find the latest model...")
        objects = list(minio_client.list_objects(MODELS_BUCKET, recursive=True))
        
        # Filter for model files with 'best' in the name
        model_objects = [obj for obj in objects if obj.object_name.lower().endswith('.pt') and 'best' in obj.object_name.lower()]
        
        if not model_objects:
            logger.warning(f"No best model found in {MODELS_BUCKET}. Using fallback model.")
            return None, YOLO_MODEL_PATH
        
        # Sort by last_modified to get the latest model
        latest_model = sorted(model_objects, key=lambda obj: obj.last_modified, reverse=True)[0]
        logger.info(f"Found latest model in MinIO: {latest_model.object_name} (modified: {latest_model.last_modified})")
        
        # Download the model to a temporary file
        _, temp_file_path = tempfile.mkstemp(suffix='.pt')
        minio_client.fget_object(MODELS_BUCKET, latest_model.object_name, temp_file_path)
        logger.info(f"Downloaded latest model to {temp_file_path}")
        
        return latest_model.object_name, temp_file_path
    except Exception as e:
        logger.error(f"Error getting latest model: {e}")
        logger.warning("Using fallback model path.")
        return None, YOLO_MODEL_PATH

def update_manifest_with_detection(client, file_identifier, max_confidence):
    """
    Update the manifest CSV on MinIO for the image identified by file_identifier,
    storing the max detection confidence in a new column 'max_confidence'.
    """
    try:
        response = client.get_object(BUCKET_NAME, MANIFEST_FILENAME)
        data = response.read()
        manifest_df = pd.read_csv(io.BytesIO(data), dtype=str)
    except Exception:
        manifest_df = pd.DataFrame(columns=["pid", "vid", "minio_object", "max_confidence"])
    
    matching = manifest_df['minio_object'].str.endswith(file_identifier)
    if matching.any():
        manifest_df.loc[matching, 'max_confidence'] = max_confidence
        logger.info(f"Updated manifest entry for {file_identifier} with max_confidence: {max_confidence}")
    else:
        # If the file is not present, create a new entry and concatenate with the existing DataFrame
        new_row = pd.DataFrame({"pid": [None], "vid": [None], "minio_object": [file_identifier], "max_confidence": [max_confidence]})
        manifest_df = pd.concat([manifest_df, new_row], ignore_index=True)
        logger.info(f"Appended new manifest entry for {file_identifier} with max_confidence: {max_confidence}")
    
    csv_data = manifest_df.to_csv(index=False)
    csv_bytes = csv_data.encode('utf-8')
    client.put_object(BUCKET_NAME, MANIFEST_FILENAME, io.BytesIO(csv_bytes), len(csv_bytes))
    logger.info(f"Manifest updated and uploaded as '{MANIFEST_FILENAME}'.")

def process_image(client, file_name, model):
    """
    Process a single image:
      - Downloads the image from MinIO.
      - Runs YOLO inference.
      - Draws detections, computes max detection confidence.
      - Uploads the detection image.
      - Updates the manifest CSV.
    """
    local_file_path = f"/tmp/temp_{os.path.basename(file_name)}"
    detection_file_path = f"/tmp/detection_{os.path.basename(file_name)}"
    try:
        client.fget_object(BUCKET_NAME, file_name, local_file_path)
        img = cv2.imread(local_file_path)
        if img is None:
            logger.warning(f"Failed to load image: {file_name}")
            return
        
        results = model(img)
        confidences = []
        for result in results:
            for box in result.boxes:
                conf = box.conf[0].item()
                confidences.append(conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = result.names[int(box.cls[0].item())]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        max_confidence = max(confidences) if confidences else 0
        cv2.imwrite(detection_file_path, img)
        detection_path = DETECTION_RESULTS_FOLDER + os.path.basename(file_name)
        with open(detection_file_path, "rb") as file_data:
            file_stat = os.stat(detection_file_path)
            client.put_object(
                BUCKET_NAME,
                detection_path,
                file_data,
                file_stat.st_size,
                content_type="image/jpeg"
            )
        logger.info(f"Processed {file_name} with max confidence: {max_confidence}")
        update_manifest_with_detection(client, os.path.basename(file_name), max_confidence)
        
    except Exception as e:
        logger.error(f"Error processing image {file_name}: {e}")
    finally:
        if os.path.exists(local_file_path):
            os.remove(local_file_path)
        if os.path.exists(detection_file_path):
            os.remove(detection_file_path)

def process_images():
    """
    List images in the inference_images folder, process each image with YOLO,
    and update the detection results in the manifest CSV.
    """
    temp_model_path = None
    try:
        # Ensure MLflow is installed
        try:
            import mlflow
        except ImportError:
            logger.info("Installing MLflow package...")
            import subprocess
            import sys
            subprocess.run([sys.executable, "-m", "pip", "install", "mlflow"], check=True)
            # Import again after installation
            import mlflow
            logger.info("MLflow installed successfully.")
            
        client = initialize_minio_client()
        
        # Get the latest model from MLflow or MinIO
        model_name, model_path = get_latest_model(client)
        temp_model_path = model_path if model_name else None
        
        logger.info(f"Loading YOLO model from {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model not found at {model_path}")
        
        model = YOLO(model_path)
        logger.info(f"YOLO model '{model_name or 'fallback'}' loaded successfully.")
        
        objects = list(client.list_objects(BUCKET_NAME, prefix=INFERENCE_IMAGES_FOLDER))
        logger.info(f"Found {len(objects)} objects in '{INFERENCE_IMAGES_FOLDER}'")
        processed_count = 0
        for obj in objects:
            if obj.object_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                process_image(client, obj.object_name, model)
                processed_count += 1
        logger.info(f"Processed {processed_count} images.")
    except Exception as e:
        logger.error(f"Error in process_images: {e}")
        raise
    finally:
        # Clean up temporary model file
        if temp_model_path and os.path.exists(temp_model_path) and temp_model_path != YOLO_MODEL_PATH:
            try:
                os.remove(temp_model_path)
                logger.info(f"Removed temporary model file {temp_model_path}")
            except Exception as e:
                logger.error(f"Error removing temporary model file: {e}")
                
        # Clean up any MLflow download directories
        if 'model_name' in locals() and model_name and model_name.startswith('mlflow:'):
            try:
                # The model_path parent directory likely contains MLflow artifacts
                mlflow_dir = os.path.dirname(model_path)
                if os.path.exists(mlflow_dir) and os.path.isdir(mlflow_dir):
                    shutil.rmtree(mlflow_dir)
                    logger.info(f"Removed MLflow artifacts directory {mlflow_dir}")
            except Exception as e:
                logger.error(f"Error removing MLflow artifacts: {e}")

# Create the Airflow DAG
dag = DAG(
    'batch_detection',
    default_args=default_args,
    description='Process images with YOLO and update detection results in manifest',
    schedule_interval=timedelta(minutes=60),
    catchup=False,
    tags=['yolo', 'minio']
)

process_task = PythonOperator(
    task_id='process_images',
    python_callable=process_images,
    dag=dag,
)

process_task

if __name__ == "__main__":
    # Allow local testing of the detection process.
    process_images()
