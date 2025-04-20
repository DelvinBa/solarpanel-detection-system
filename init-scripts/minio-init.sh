#!/bin/bash
set -e

# Wait for MinIO to be ready
echo "Waiting for MinIO to be ready..."
until curl -s http://minio:9000/minio/health/live; do
  echo "MinIO not ready yet, waiting..."
  sleep 5
done

echo "MinIO is ready. Creating buckets and uploading data..."

# Create a Python script to initialize buckets and upload data
cat > /tmp/create_buckets.py << 'EOF'
from minio import Minio
from minio.error import S3Error
import os
import io
import shutil
import glob
from pathlib import Path

def get_minio_client():
    """
    Returns a Minio client instance configured for the local setup.
    """
    endpoint = "minio:9000"  
    access_key = "minioadmin"
    secret_key = "minioadmin"
    secure = False
    client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
    return client

def init_minio_buckets():
    """
    Initializes MinIO by creating the required buckets:
    1. inference-data
    2. mlflow
    """
    client = get_minio_client()

    # Define buckets and their subfolders
    buckets_and_folders = {
        "inference-data": ["inference_images/", "detection_results/"],
        "mlflow": []  # No specific subfolders for mlflow
    }

    for bucket, folders in buckets_and_folders.items():
        # Create the bucket if it doesn't exist
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
            print(f"Bucket '{bucket}' created.")
        else:
            print(f"Bucket '{bucket}' already exists.")

        # Create subfolders
        for folder in folders:
            try:
                objects = list(client.list_objects(bucket, prefix=folder, recursive=True))
                if not objects:
                    # Create an empty object with the folder name to simulate a folder
                    client.put_object(bucket, folder, io.BytesIO(b""), 0)
                    print(f"Folder '{folder}' created in bucket '{bucket}'.")
                else:
                    print(f"Folder '{folder}' already exists in bucket '{bucket}'.")
            except Exception as e:
                print(f"Error creating folder {folder}: {str(e)}")

def upload_data_to_mlflow():
    """
    Upload all data from /solarpanel_detection_service/data to the mlflow bucket
    """
    client = get_minio_client()
    data_dir = "/solarpanel_detection_service/data"
    bucket_name = "mlflow"
    
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist. Skipping upload.")
        return
    
    # Make sure mlflow bucket exists
    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)
        print(f"Bucket '{bucket_name}' created for MLflow data.")
    
    # Walk through all files in the data directory
    file_count = 0
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            # Get the full file path
            file_path = os.path.join(root, file)
            
            # Calculate the object name by removing the data directory prefix
            object_name = os.path.relpath(file_path, data_dir)
            
            # Upload the file to MinIO
            try:
                file_size = os.path.getsize(file_path)
                with open(file_path, 'rb') as file_data:
                    client.put_object(
                        bucket_name,
                        f"data/{object_name}",  # Put everything under a data/ prefix
                        file_data,
                        file_size
                    )
                file_count += 1
                if file_count % 100 == 0:
                    print(f"Uploaded {file_count} files so far...")
            except Exception as e:
                print(f"Error uploading file {file_path}: {str(e)}")
    
    print(f"Uploaded {file_count} files to the MLflow bucket.")

if __name__ == "__main__":
    try:
        print("Initializing MinIO buckets...")
        init_minio_buckets()
        
        print("Uploading data to MLflow bucket...")
        upload_data_to_mlflow()
        
        print("MinIO initialization completed successfully.")
    except S3Error as err:
        print(f"Error initializing MinIO: {err}")
EOF

# Install required packages for the initialization script
pip install minio

# Run the Python script to create buckets and upload data
python /tmp/create_buckets.py

echo "MinIO initialization completed."