from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from minio import Minio
import cv2
import os
import logging
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use environment variables for configuration
MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', '172.19.0.1')
MINIO_PORT = os.getenv('MINIO_PORT', '9000')
MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY', 'minioadmin')
BUCKET_NAME = os.getenv('MINIO_BUCKET', 'mybucket')
LABELED_FOLDER = "labeled-images/"
ORIGINAL_FOLDER = "original-files/"
# Get the absolute path to the DAGs folder
DAGS_FOLDER = os.path.dirname(os.path.abspath(__file__))
# Path to the YOLO model inside the Airflow container
YOLO_MODEL_PATH = os.path.join(DAGS_FOLDER, "models", "best5.pt")

# DAG Default Arguments
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
    'pool': 'default_pool'
}

def initialize_minio_client():
    """Initialize and return MinIO client."""
    try:
        logger.info(f"Attempting to connect to MinIO at {MINIO_ENDPOINT}:{MINIO_PORT}")
        # Initialize MinIO client with the correct hostname and port
        client = Minio(
            endpoint=f"{MINIO_ENDPOINT}:{MINIO_PORT}",
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=False
        )
        logger.info(f"Successfully initialized MinIO client with endpoint: {MINIO_ENDPOINT}:{MINIO_PORT}")
        
        # Test connection by listing buckets
        logger.info("Testing MinIO connection by listing buckets...")
        try:
            buckets = client.list_buckets()
            bucket_names = [b.name for b in buckets]
            logger.info(f"Available buckets: {bucket_names}")
        except Exception as bucket_error:
            logger.error(f"Error listing buckets: {str(bucket_error)}")
            raise
        
        # Create bucket if it doesn't exist
        try:
            if not client.bucket_exists(BUCKET_NAME):
                logger.info(f"Bucket {BUCKET_NAME} does not exist, creating it")
                client.make_bucket(BUCKET_NAME)
                logger.info(f"Created bucket: {BUCKET_NAME}")
            else:
                logger.info(f"Bucket {BUCKET_NAME} already exists")
        except Exception as bucket_error:
            logger.error(f"Error checking/creating bucket: {str(bucket_error)}")
            raise
            
        return client
    except Exception as e:
        logger.error(f"Error initializing MinIO client: {str(e)}")
        raise

def move_to_original(client, file_name):
    """Move processed file to original-files folder."""
    try:
        # Create the original-files folder if it doesn't exist
        if not client.bucket_exists(BUCKET_NAME):
            client.make_bucket(BUCKET_NAME)
            
        # Copy the file to the original-files folder
        original_path = ORIGINAL_FOLDER + file_name
        client.copy_object(
            BUCKET_NAME,
            original_path,
            f"{BUCKET_NAME}/{file_name}"
        )
        
        # Remove the file from the root of the bucket
        client.remove_object(BUCKET_NAME, file_name)
        
        logger.info(f"Moved {file_name} to {original_path}")
    except Exception as e:
        logger.error(f"Error moving file to original folder: {str(e)}")
        raise

def process_image(client, file_name, model):
    """Process a single image with YOLO model."""
    local_file_path = f"/tmp/temp_{os.path.basename(file_name)}"
    labeled_file_path = f"/tmp/labeled_{os.path.basename(file_name)}"
    
    try:
        # Download image from MinIO
        client.fget_object(BUCKET_NAME, file_name, local_file_path)
        
        # Read image using OpenCV
        img = cv2.imread(local_file_path)
        
        if img is None:
            logger.warning(f"Failed to load image: {file_name}")
            return
            
        # Run YOLO inference
        results = model(img)
        
        # Draw results
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                label = result.names[int(box.cls[0].item())]
                
                # Draw bounding box and label
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save the labeled image locally
        cv2.imwrite(labeled_file_path, img)
        
        # Upload the labeled image to MinIO
        labeled_path = LABELED_FOLDER + file_name
        with open(labeled_file_path, "rb") as file_data:
            file_stat = os.stat(labeled_file_path)
            client.put_object(
                BUCKET_NAME,
                labeled_path,
                file_data,
                file_stat.st_size,
                content_type="image/jpeg"
            )
        
        # Move the original file to original-files folder
        move_to_original(client, file_name)
        
        logger.info(f"Successfully processed image: {file_name}")
        logger.info(f"Labeled image saved to MinIO: {labeled_path}")
        
    except Exception as img_error:
        logger.error(f"Error processing image {file_name}: {img_error}")
    finally:
        # Cleanup local files
        if os.path.exists(local_file_path):
            os.remove(local_file_path)
        if os.path.exists(labeled_file_path):
            os.remove(labeled_file_path)

def process_images():
    """Download images from MinIO, run YOLO inference, and upload labeled images back to MinIO."""
    try:
        # Initialize MinIO client
        logger.info("Initializing MinIO client...")
        client = initialize_minio_client()
        
        # Initialize YOLO Model
        logger.info(f"Loading YOLO model from {YOLO_MODEL_PATH}")
        if not os.path.exists(YOLO_MODEL_PATH):
            logger.error(f"YOLO model not found at {YOLO_MODEL_PATH}")
            raise FileNotFoundError(f"YOLO model not found at {YOLO_MODEL_PATH}")
            
        model = YOLO(YOLO_MODEL_PATH)
        logger.info(f"YOLO model loaded successfully")
        
        # List objects in the bucket
        logger.info(f"Listing objects in bucket: {BUCKET_NAME}")
        objects = list(client.list_objects(BUCKET_NAME))
        logger.info(f"Found {len(objects)} objects in bucket")
        
        # Log all object names for debugging
        for obj in objects:
            logger.info(f"Found object: {obj.object_name}")
        
        # Process each image
        image_count = 0
        for obj in objects:
            logger.info(f"Checking object: {obj.object_name}")
            if obj.object_name.endswith(('.jpg', '.jpeg', '.png')) and not obj.object_name.startswith((LABELED_FOLDER, ORIGINAL_FOLDER)):
                logger.info(f"Processing image: {obj.object_name}")
                process_image(client, obj.object_name, model)
                image_count += 1
            else:
                logger.info(f"Skipping object {obj.object_name} - not an image or already processed")
        
        logger.info(f"Processed {image_count} images")
                
    except Exception as e:
        logger.error(f"Error in process_images: {str(e)}")
        raise

# Create the DAG
dag = DAG(
    'yolo_minio_airflow',
    default_args=default_args,
    description='Process images using YOLO model and store results in MinIO',
    schedule_interval=timedelta(minutes=5),
    catchup=False,
    tags=['yolo', 'minio'],
)

# Create the task
process_task = PythonOperator(
    task_id='process_images',
    python_callable=process_images,
    dag=dag,
)

# Set task dependencies
process_task 