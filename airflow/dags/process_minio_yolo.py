from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from minio import Minio
import cv2
import os
import logging
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Table, MetaData
from sqlalchemy.sql import text
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use environment variables for configuration
MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'http://mlflow_minio:9000')  # Updated to use correct container name in docker-compose
MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', 'RCo66DoKbvreL2ou')
MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY', 'IEQTtmE3wpJln06SH3AsW8BAqeNs0q36')
BUCKET_NAME = os.getenv('MINIO_BUCKET', 'mybucket')
LABELED_FOLDER = "labeled-images/"
ORIGINAL_FOLDER = "original-files/"  # New folder for original files

# PostgreSQL connection string
POSTGRES_CONN = "postgresql+psycopg2://airflow:airflow@airflow_postgres/airflow"

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
    'execution_timeout': timedelta(minutes=30),  # Increased timeout
    'max_active_runs': 1,  # Limit concurrent runs
    'max_active_tasks': 1,  # Limit concurrent tasks
    'pool': 'default_pool'  # Use default pool for better resource management
}

def create_image_logs_table():
    """Create the image_processing_logs table if it doesn't exist."""
    try:
        engine = create_engine(POSTGRES_CONN)
        metadata = MetaData()
        
        Table('image_processing_logs', metadata,
            Column('id', Integer, primary_key=True),
            Column('image_name', String),
            Column('processed_at', DateTime),
            Column('labeled_path', String),
            Column('num_detections', Integer),
            Column('avg_confidence', Float),
            Column('status', String),
            Column('error_message', String, nullable=True)
        )
        
        metadata.create_all(engine)
        logger.info("Image logs table created or already exists")
    except Exception as e:
        logger.error(f"Error creating logs table: {str(e)}")
        raise

def log_to_postgres(image_name, labeled_path, num_detections, avg_confidence, status, error_message=None):
    """Log image processing results to PostgreSQL."""
    try:
        engine = create_engine(POSTGRES_CONN)
        
        with engine.connect() as connection:
            connection.execute(text("""
                INSERT INTO image_processing_logs 
                (image_name, processed_at, labeled_path, num_detections, avg_confidence, status, error_message)
                VALUES (:image_name, :processed_at, :labeled_path, :num_detections, :avg_confidence, :status, :error_message)
            """), {
                'image_name': image_name,
                'processed_at': datetime.now(),
                'labeled_path': labeled_path,
                'num_detections': num_detections,
                'avg_confidence': avg_confidence,
                'status': status,
                'error_message': error_message
            })
        logger.info(f"Logged processing of image {image_name} with status {status}")
    except Exception as e:
        logger.error(f"Error logging to Postgres: {str(e)}")

def initialize_minio_client():
    """Initialize and return MinIO client.111"""
    try:
        # Remove any protocol prefix and ensure clean endpoint
        endpoint = MINIO_ENDPOINT.replace('http://', '').replace('https://', '')
        
        # Split endpoint into host and port if needed
        if ':' in endpoint:
            host, port = endpoint.split(':')
            endpoint = host
        
        client = Minio(
            endpoint=endpoint,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=False
        )
        
        # Test connection
        client.list_buckets()
        
        # Create bucket if it doesn't exist
        if not client.bucket_exists(BUCKET_NAME):
            logger.info(f"Bucket {BUCKET_NAME} does not exist, creating it")
            client.make_bucket(BUCKET_NAME)
            
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
            log_to_postgres(
                image_name=file_name,
                labeled_path="",
                num_detections=0,
                avg_confidence=0.0,
                status="FAILED",
                error_message="Failed to load image"
            )
            return
            
        # Run YOLO inference
        results = model(img)
        
        # Initialize detection stats
        num_detections = 0
        total_confidence = 0.0
        
        # Draw results
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                label = result.names[int(box.cls[0].item())]
                
                # Update detection stats
                num_detections += 1
                total_confidence += conf
                
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
        
        # Log successful processing to PostgreSQL
        avg_confidence = total_confidence / num_detections if num_detections > 0 else 0.0
        log_to_postgres(
            image_name=file_name,
            labeled_path=labeled_path,
            num_detections=num_detections,
            avg_confidence=avg_confidence,
            status="SUCCESS"
        )
        
        logger.info(f"Labeled image saved to MinIO: {labeled_path}")
        
    except Exception as img_error:
        logger.error(f"Error processing image {file_name}: {img_error}")
        log_to_postgres(
            image_name=file_name,
            labeled_path="",
            num_detections=0,
            avg_confidence=0.0,
            status="ERROR",
            error_message=str(img_error)
        )
    finally:
        # Cleanup local files
        if os.path.exists(local_file_path):
            os.remove(local_file_path)
        if os.path.exists(labeled_file_path):
            os.remove(labeled_file_path)

def process_images():
    """Download images from MinIO, run YOLO inference, and upload labeled images back to MinIO."""
    try:
        # Create the image logs table if it doesn't exist
        create_image_logs_table()
        
        # Initialize MinIO client
        client = initialize_minio_client()
        
        # Initialize YOLO Model
        if not os.path.exists(YOLO_MODEL_PATH):
            logger.error(f"YOLO model not found at {YOLO_MODEL_PATH}")
            raise FileNotFoundError(f"YOLO model not found at {YOLO_MODEL_PATH}")
            
        model = YOLO(YOLO_MODEL_PATH)
        logger.info(f"YOLO model loaded from {YOLO_MODEL_PATH}")
        
        # List objects in the bucket
        objects = client.list_objects(BUCKET_NAME, recursive=True)
        
        # Process each object
        for obj in objects:
            file_name = obj.object_name
            
            # Skip already labeled images
            if file_name.startswith(LABELED_FOLDER):
                continue
                
            process_image(client, file_name, model)
            
        logger.info("Image processing complete.")
        
    except Exception as e:
        logger.error(f"Error in image processing workflow: {e}")
        raise

# Define the DAG
dag = DAG(
    'yolo_minio_airflow',
    default_args=default_args,
    description='Process images using YOLO model and store results in MinIO',
    schedule_interval='*/10 * * * *',  # Run every 10 minutes
    catchup=False,
    tags=['yolo', 'minio'],
    max_active_runs=1,  # Limit concurrent runs
    concurrency=1  # Limit concurrent tasks
)

# Define the task
process_images = PythonOperator(
    task_id='process_images',
    python_callable=process_images,
    dag=dag,
    execution_timeout=timedelta(minutes=30),  # Increased timeout
    pool='default_pool'  # Use default pool for better resource management
)

process_images  # Execution order (only one task in this case) 