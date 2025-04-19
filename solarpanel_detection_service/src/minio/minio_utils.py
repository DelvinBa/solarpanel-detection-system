from minio import Minio
import io

def get_minio_client():
    """
    Returns a Minio client instance.
    """
    endpoint = "minio:9000"  
    access_key = "minioadmin"
    secret_key = "minioadmin"
    secure = False
    client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
    return client

def upload_image_to_minio(image, object_name, bucket_name, minio_client=None):
    """
    Uploads a PIL Image to MinIO and returns the object name.
    
    :param image: PIL.Image object to upload.
    :param object_name: The object name/key in the bucket (e.g., 'training_images/0153010000442005.jpg').
    :param bucket_name: The MinIO bucket name (e.g. 'training_data').
    :param minio_client: Optional pre-created Minio client.
    :return: The object name if upload is successful.
    """
    if minio_client is None:
        minio_client = get_minio_client()
    
    # Convert image to bytes
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")
    image_bytes.seek(0)
    file_size = image_bytes.getbuffer().nbytes

    # Ensure bucket exists; if not, create it
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)
    
    # Upload the image
    minio_client.put_object(
        bucket_name,
        object_name,
        image_bytes,
        file_size,
        content_type="image/jpeg"
    )
    print(f"Uploaded image to MinIO: {bucket_name}/{object_name}")
    return object_name
