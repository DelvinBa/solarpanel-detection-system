from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from minio import Minio
import cv2
import os
from ultralytics import YOLO
import logging
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Table, MetaData
from sqlalchemy.sql import text

# Use environment variables for configuration
MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'http://minio:9000')
MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY', 'minioadmin')
BUCKET_NAME = os.getenv('MINIO_BUCKET', 'mybucket')
LABELED_FOLDER = "labeled-images/"

# PostgreSQL connection string
POSTGRES_CONN = "postgresql+psycopg2://airflow:airflow@postgres/airflow"

# Get the absolute path to the DAGs folder
DAGS_FOLDER = os.path.dirname(os.path.abspath(__file__))

# Path to the YOLO model inside the Airflow container
YOLO_MODEL_PATH = os.path.join(DAGS_FOLDER, "models", "best5.pt")

# DAG Default Arguments
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2024, 2, 24),
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_image_logs_table():
    """Create the image_processing_logs table if it doesn't exist."""
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

def log_to_postgres(image_name, labeled_path, num_detections, avg_confidence, status, error_message=None):
    """Log image processing results to PostgreSQL."""
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

def process_images():
    """Download images from MinIO, run YOLO inference, and upload labeled images back to MinIO."""
    try:
        # Create the image logs table if it doesn't exist
        create_image_logs_table()
        
        # Initialize MinIO client
        client = Minio(
            endpoint=MINIO_ENDPOINT.replace("http://", "").replace("https://", ""),
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=False
        )

        # Verify bucket exists
        if not client.bucket_exists(BUCKET_NAME):
            logger.error(f"Bucket {BUCKET_NAME} does not exist")
            return

        # Initialize YOLO Model
        model = YOLO(YOLO_MODEL_PATH)

        # List objects in the bucket
        objects = client.list_objects(BUCKET_NAME, recursive=True)

        # Process each object
        for obj in objects:
            try:
                file_name = obj.object_name
                local_file_path = f"/tmp/temp_{file_name}"
                labeled_file_path = f"/tmp/labeled_{file_name}"

                # Skip already labeled images
                if file_name.startswith(LABELED_FOLDER):
                    continue

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
                    continue

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

                # Cleanup local files
                os.remove(local_file_path)
                os.remove(labeled_file_path)

            except Exception as img_error:
                logger.error(f"Error processing image {obj.object_name}: {img_error}")
                log_to_postgres(
                    image_name=obj.object_name,
                    labeled_path="",
                    num_detections=0,
                    avg_confidence=0.0,
                    status="ERROR",
                    error_message=str(img_error)
                )
                continue

        logger.info("Image processing complete.")

    except Exception as e:
        logger.error(f"Error in image processing workflow: {e}")
        raise

# Define DAG
with DAG(
    dag_id="yolo_minio_airflow",
    default_args=default_args,
    schedule_interval="*/2 * * * *",  # Runs every 2 minutes
    catchup=False,
    tags=["minio", "yolo", "image-processing"],
) as dag:

    process_task = PythonOperator(
        task_id="process_images",
        python_callable=process_images
    )

    process_task  # Execution order (only one task in this case)