import os
import cv2
import logging
import pandas as pd
import io
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from minio import Minio
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment configuration
MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 's3')
MINIO_PORT = os.getenv('MINIO_PORT', '9000')
MINIO_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID', os.getenv('MINIO_ACCESS_KEY', 'minioadmin'))
MINIO_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', os.getenv('MINIO_SECRET_KEY', 'minioadmin'))
BUCKET_NAME = os.getenv('MINIO_BUCKET', 'inference-data')
MINIO_SECURE = os.getenv('MINIO_SECURE', 'False').lower() == 'true'
INFERENCE_IMAGES_FOLDER = "inference_images/"
DETECTION_RESULTS_FOLDER = "detection_results/"
MANIFEST_FILENAME = "house_id_results.csv"

# Path to YOLO model
DAGS_FOLDER = os.path.dirname(os.path.abspath(__file__))
YOLO_MODEL_PATH = os.path.join(DAGS_FOLDER, "models", "best5.pt")

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
        # If port is specified and not using AWS S3, include it in the endpoint
        if MINIO_PORT != '80' and MINIO_PORT != '443' and not (MINIO_ENDPOINT.startswith('s3.') or MINIO_ENDPOINT == 's3'):
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
        if not client.bucket_exists(BUCKET_NAME):
            client.make_bucket(BUCKET_NAME)
            logger.info(f"Bucket '{BUCKET_NAME}' created.")
        return client
    except Exception as e:
        logger.error(f"Error initializing MinIO client: {e}")
        raise

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
        # If the file is not present, optionally append a new entry.
        new_row = {"pid": None, "vid": None, "minio_object": file_identifier, "max_confidence": max_confidence}
        manifest_df = manifest_df.append(new_row, ignore_index=True)
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
    try:
        client = initialize_minio_client()
        if not os.path.exists(YOLO_MODEL_PATH):
            raise FileNotFoundError(f"YOLO model not found at {YOLO_MODEL_PATH}")
        model = YOLO(YOLO_MODEL_PATH)
        logger.info("YOLO model loaded successfully.")
        
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

# Create the Airflow DAG
dag = DAG(
    'yolo_minio_airflow',
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
