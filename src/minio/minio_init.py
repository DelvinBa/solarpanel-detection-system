from minio import Minio
from minio.error import S3Error
import io

def get_minio_client():
    """
    Returns a Minio client instance configured for your local setup.
    Adjust the endpoint, access key, and secret key as needed.
    """
    endpoint = "localhost:9000"  # or "http://localhost:9000" if you prefer
    access_key = "minioadmin"
    secret_key = "minioadmin"
    secure = False
    client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
    return client

def init_minio_buckets():
    """
    Initializes MinIO by creating the necessary buckets and subfolder structure
    for the entire project.
    """
    client = get_minio_client()

    # Buckets and their subfolders
    # (Folders are simulated in S3/MinIO by creating empty objects with a trailing slash)
    buckets_and_folders = {
        "training-data":   ["training_images/", "labels/"],
        "inference-data":  ["inference_images/", "detection_results/"],
        "models":          []  # Or [] if you don't need subfolders here
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
            objects = list(client.list_objects(bucket, prefix=folder, recursive=True))
            if not objects:
                # Create an empty object with the folder name to simulate a folder
                client.put_object(bucket, folder, io.BytesIO(b""), 0)
                print(f"Folder '{folder}' created in bucket '{bucket}'.")
            else:
                print(f"Folder '{folder}' already exists in bucket '{bucket}'.")

if __name__ == "__main__":
    try:
        init_minio_buckets()
    except S3Error as err:
        print("Error initializing MinIO:", err)
