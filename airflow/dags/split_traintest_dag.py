"""
Solar Panel Detection Train-Test-Val Split DAG
=============================================

This DAG performs the train/test/validation split for the solar panel detection dataset.
It reads raw data from MinIO and writes the split data back to MinIO.

"""

from datetime import datetime, timedelta
import os
import shutil
import random
import tempfile
from pathlib import Path
import logging

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from minio import Minio
from minio.error import S3Error

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    'split_traintest_solar_panel',
    default_args=default_args,
    description='Split solar panel dataset into train/test/validation sets',
    schedule_interval=None,  # Set to None for manual triggering
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['solar-panel', 'data-preparation', 'train-test-split'],
)

# MinIO configuration
MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 's3')
MINIO_PORT = os.getenv('MINIO_PORT', '9000')
MINIO_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID', os.getenv('MINIO_ACCESS_KEY', 'minioadmin'))
MINIO_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', os.getenv('MINIO_SECRET_KEY', 'minioadmin'))
MINIO_BUCKET = os.getenv('MINIO_BUCKET', 'mlflow')
MINIO_SECURE = os.getenv('MINIO_SECURE', 'False').lower() == 'true'

# Source and destination paths in MinIO
SOURCE_PREFIX = "data/raw/"
DESTINATION_PREFIX = "data/processed/SateliteData/"

# Temp directory for processing
TEMP_DIR = Variable.get('split_temp_dir', default_var='/tmp/solar_panel_split')

# Train/val/test split ratios
TRAIN_RATIO = float(Variable.get('train_ratio', default_var=0.8))
VAL_RATIO = float(Variable.get('val_ratio', default_var=0.1))
# Test ratio is 1 - (TRAIN_RATIO + VAL_RATIO)

def initialize_minio_client():
    """Initialize and return MinIO client."""
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
        logger.info(f"Successfully initialized MinIO/S3 client with endpoint: {endpoint}")
        
        # Test connection by listing buckets
        logger.info("Testing MinIO/S3 connection by listing buckets...")
        try:
            buckets = client.list_buckets()
            bucket_names = [b.name for b in buckets]
            logger.info(f"Available buckets: {bucket_names}")
        except Exception as bucket_error:
            logger.error(f"Error listing buckets: {str(bucket_error)}")
            raise
        
        # Check if our bucket exists
        try:
            if not client.bucket_exists(MINIO_BUCKET):
                logger.warning(f"Bucket {MINIO_BUCKET} does not exist!")
                raise ValueError(f"Bucket {MINIO_BUCKET} not found")
            else:
                logger.info(f"Bucket {MINIO_BUCKET} exists")
        except Exception as bucket_error:
            logger.error(f"Error checking bucket: {str(bucket_error)}")
            raise
            
        return client
    except Exception as e:
        logger.error(f"Error initializing MinIO/S3 client: {str(e)}")
        raise

def download_raw_data(**kwargs):
    """Download raw data from MinIO to local temp directory."""
    # Initialize MinIO client
    client = initialize_minio_client()
    
    # Create temp directory structure
    temp_dir = Path(TEMP_DIR)
    if temp_dir.exists():
        logger.info(f"Cleaning existing temp directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    logger.info(f"Creating temp directory structure: {temp_dir}")
    
    # Create raw data directories
    raw_dir = temp_dir / 'raw'
    images_dir = raw_dir / 'images'
    labels_dir = raw_dir / 'labels'
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # List objects in MinIO raw directory
    logger.info(f"Listing objects in MinIO bucket: {MINIO_BUCKET}, prefix: {SOURCE_PREFIX}")
    
    # Try to find folders or files that might contain images and labels
    all_objects = list(client.list_objects(MINIO_BUCKET, prefix=SOURCE_PREFIX, recursive=True))
    
    # Filter for image and label files
    image_objects = [obj for obj in all_objects if any(obj.object_name.lower().endswith(ext) 
                                                   for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff'])]
    label_objects = [obj for obj in all_objects if obj.object_name.lower().endswith('.txt')]
    
    logger.info(f"Found {len(image_objects)} image files and {len(label_objects)} label files in MinIO")
    
    if not image_objects:
        logger.error("No image files found in MinIO. Cannot proceed with split.")
        raise ValueError("No image files found in MinIO source location")
    
    # Download images
    logger.info(f"Downloading {len(image_objects)} image files to {images_dir}")
    downloaded_images = 0
    for obj in image_objects:
        try:
            # Get filename from object path
            filename = os.path.basename(obj.object_name)
            local_path = images_dir / filename
            
            # Download file
            client.fget_object(MINIO_BUCKET, obj.object_name, str(local_path))
            downloaded_images += 1
            
            if downloaded_images % 10 == 0:
                logger.info(f"Downloaded {downloaded_images}/{len(image_objects)} image files")
        except Exception as e:
            logger.error(f"Error downloading image {obj.object_name}: {str(e)}")
    
    # Download labels
    logger.info(f"Downloading {len(label_objects)} label files to {labels_dir}")
    downloaded_labels = 0
    for obj in label_objects:
        try:
            # Get filename from object path
            filename = os.path.basename(obj.object_name)
            local_path = labels_dir / filename
            
            # Download file
            client.fget_object(MINIO_BUCKET, obj.object_name, str(local_path))
            downloaded_labels += 1
            
            if downloaded_labels % 10 == 0:
                logger.info(f"Downloaded {downloaded_labels}/{len(label_objects)} label files")
        except Exception as e:
            logger.error(f"Error downloading label {obj.object_name}: {str(e)}")
    
    logger.info(f"Successfully downloaded {downloaded_images} image files and {downloaded_labels} label files")
    
    # Push paths to XCom for next task
    kwargs['ti'].xcom_push(key='temp_dir', value=str(temp_dir))
    kwargs['ti'].xcom_push(key='images_dir', value=str(images_dir))
    kwargs['ti'].xcom_push(key='labels_dir', value=str(labels_dir))
    
    return {
        'temp_dir': str(temp_dir),
        'images_dir': str(images_dir),
        'labels_dir': str(labels_dir),
        'image_count': downloaded_images,
        'label_count': downloaded_labels
    }

def split_data(**kwargs):
    """Split the dataset into train/validation/test sets."""
    ti = kwargs['ti']
    
    # Get directories from previous task
    temp_dir = Path(ti.xcom_pull(task_ids='download_raw_data', key='temp_dir'))
    images_dir = Path(ti.xcom_pull(task_ids='download_raw_data', key='images_dir'))
    labels_dir = Path(ti.xcom_pull(task_ids='download_raw_data', key='labels_dir'))
    
    logger.info(f"Setting up split with data from: {images_dir} and {labels_dir}")
    
    # Create output directories - Remove the redundant SateliteData from the path
    processed_dir = temp_dir / 'processed'
    
    train_images_path = processed_dir / 'train' / 'images'
    train_labels_path = processed_dir / 'train' / 'labels'
    val_images_path = processed_dir / 'val' / 'images'
    val_labels_path = processed_dir / 'val' / 'labels'
    test_images_path = processed_dir / 'test' / 'images'
    test_labels_path = processed_dir / 'test' / 'labels'
    
    # Create directory structure
    for dir_path in [train_images_path, train_labels_path, val_images_path, 
                     val_labels_path, test_images_path, test_labels_path]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # List all image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
        image_files.extend([f for f in os.listdir(images_dir) if f.lower().endswith(ext)])
    
    logger.info(f"Found {len(image_files)} images for splitting")
    
    if len(image_files) == 0:
        logger.error("No image files found for splitting")
        raise ValueError("No image files found for splitting")
    
    # Shuffle images for randomization
    random.shuffle(image_files)
    
    # Calculate split points
    train_split = int(TRAIN_RATIO * len(image_files))
    val_split = int((TRAIN_RATIO + VAL_RATIO) * len(image_files))
    
    # Split the dataset
    train_files = image_files[:train_split]
    val_files = image_files[train_split:val_split]
    test_files = image_files[val_split:]
    
    logger.info(f"Split: {len(train_files)} training, {len(val_files)} validation, {len(test_files)} test images")
    
    # Copy function
    def copy_files(file_list, src_image_path, src_label_path, dest_image_path, dest_label_path):
        copied_images = 0
        copied_labels = 0
        
        for file_name in file_list:
            # Copy image
            src_img = src_image_path / file_name
            dst_img = dest_image_path / file_name
            
            if src_img.exists():
                shutil.copy2(src_img, dst_img)
                copied_images += 1
            else:
                logger.warning(f"Source image not found: {src_img}")
            
            # Try to find and copy corresponding label
            base_name = os.path.splitext(file_name)[0]
            for label_ext in ['.txt']:
                label_name = f"{base_name}{label_ext}"
                src_label = src_label_path / label_name
                
                if src_label.exists():
                    dst_label = dest_label_path / label_name
                    shutil.copy2(src_label, dst_label)
                    copied_labels += 1
                    break
        
        return copied_images, copied_labels
    
    # Copy files to respective directories
    logger.info("Copying training files...")
    train_stats = copy_files(train_files, images_dir, labels_dir, train_images_path, train_labels_path)
    
    logger.info("Copying validation files...")
    val_stats = copy_files(val_files, images_dir, labels_dir, val_images_path, val_labels_path)
    
    logger.info("Copying test files...")
    test_stats = copy_files(test_files, images_dir, labels_dir, test_images_path, test_labels_path)
    
    logger.info(f"Training: {train_stats[0]} images, {train_stats[1]} labels")
    logger.info(f"Validation: {val_stats[0]} images, {val_stats[1]} labels")
    logger.info(f"Test: {test_stats[0]} images, {test_stats[1]} labels")
    
    # Push processed directory to XCom
    kwargs['ti'].xcom_push(key='processed_dir', value=str(processed_dir))
    
    return {
        'processed_dir': str(processed_dir),
        'train_count': train_stats[0],
        'val_count': val_stats[0],
        'test_count': test_stats[0]
    }

def upload_to_minio(**kwargs):
    """Upload split data back to MinIO."""
    ti = kwargs['ti']
    
    # Get processed directory from previous task
    processed_dir = Path(ti.xcom_pull(task_ids='split_data', key='processed_dir'))
    
    logger.info(f"Uploading split data from {processed_dir} to MinIO")
    
    # Initialize MinIO client
    client = initialize_minio_client()
    
    # Count for tracking progress
    uploaded_files = 0
    
    # Walk through the processed directory
    for root, dirs, files in os.walk(processed_dir):
        for file in files:
            # Get the full local path
            local_path = os.path.join(root, file)
            
            # Calculate the object name in MinIO
            rel_path = os.path.relpath(local_path, start=processed_dir)
            object_name = os.path.join(DESTINATION_PREFIX, rel_path).replace('\\', '/')
            
            try:
                # Upload the file to MinIO
                client.fput_object(
                    bucket_name=MINIO_BUCKET,
                    object_name=object_name,
                    file_path=local_path
                )
                uploaded_files += 1
                
                if uploaded_files % 10 == 0:
                    logger.info(f"Uploaded {uploaded_files} files to MinIO")
            except Exception as e:
                logger.error(f"Error uploading {local_path} to {object_name}: {str(e)}")
    
    logger.info(f"Successfully uploaded {uploaded_files} files to MinIO bucket {MINIO_BUCKET}")
    
    # Clean up temp directory
    try:
        shutil.rmtree(processed_dir.parent.parent)  # Remove the entire temp directory
        logger.info(f"Cleaned up temporary directory {processed_dir.parent.parent}")
    except Exception as e:
        logger.warning(f"Error cleaning up temporary directory: {str(e)}")
    
    return uploaded_files

# Define the tasks in the DAG
with dag:
    # Task to download raw data from MinIO
    download_data_task = PythonOperator(
        task_id='download_raw_data',
        python_callable=download_raw_data,
        provide_context=True,
    )
    
    # Task to split the data
    split_data_task = PythonOperator(
        task_id='split_data',
        python_callable=split_data,
        provide_context=True,
    )
    
    # Task to upload split data back to MinIO
    upload_data_task = PythonOperator(
        task_id='upload_to_minio',
        python_callable=upload_to_minio,
        provide_context=True,
    )
    
    # Define task dependencies
    download_data_task >> split_data_task >> upload_data_task 