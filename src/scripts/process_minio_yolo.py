from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from minio import Minio
import cv2
import os
from ultralytics import YOLO
import logging

# Use environment variables for configuration
MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'http://minio:9000')
MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY', 'minioadmin')
BUCKET_NAME = os.getenv('MINIO_BUCKET', 'mybucket')
LABELED_FOLDER = "labeled-images/"

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

def process_images():
    """Download images from MinIO, run YOLO inference, and upload labeled images back to MinIO."""
    try:
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
                    continue

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
                with open(labeled_file_path, "rb") as file_data:
                    file_stat = os.stat(labeled_file_path)
                    client.put_object(
                        BUCKET_NAME,
                        LABELED_FOLDER + file_name,
                        file_data,
                        file_stat.st_size,
                        content_type="image/jpeg"
                    )

                logger.info(f"Labeled image saved to MinIO: {LABELED_FOLDER + file_name}")

                # Cleanup local files
                os.remove(local_file_path)
                os.remove(labeled_file_path)

            except Exception as img_error:
                logger.error(f"Error processing image {obj.object_name}: {img_error}")
                continue

        logger.info("Image processing complete.")

    except Exception as e:
        logger.error(f"Error in image processing workflow: {e}")
        raise

# Define DAG
with DAG(
    dag_id="yolo_minio_airflow",
    default_args=default_args,
    schedule_interval="*/10 * * * *",  # Runs every 10 minutes
    catchup=False,
    tags=["minio", "yolo", "image-processing"],
) as dag:

    process_task = PythonOperator(
        task_id="process_images",
        python_callable=process_images
    )

    process_task  # Execution order (only one task in this case)