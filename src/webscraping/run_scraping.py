import pandas as pd
from .scrape_house_ids import get_pids_and_vids
from .scrape_location import get_coordinates
from .scrape_images import get_aerial_image
from src.minio.minio_utils import upload_image_to_minio, get_minio_client

def run_pipeline(gemeentecode="GM0153", limit=10, demo=False):
    """
    Runs the scraping pipeline:
      1. Fetch all house IDs (pid and vid) for a given gemeentecode.
      2. For each record, fetch coordinates and an aerial image.
      3. Immediately upload the image to MinIO using the pid as the object name.
      4. Create a manifest CSV mapping pid and MinIO object path.
    
    :param gemeentecode: str, the city code.
    :param limit: int, maximum number of records to process.
    :param demo: bool, if True, process only a single record (for demonstration).
    """
    # Step 1: Fetch house IDs and VIDs
    df_ids = get_pids_and_vids(gemeentecode)
    if demo:
        df_ids = df_ids.head(1)
    else:
        df_ids = df_ids.head(limit)
    
    # Create a MinIO client and define the bucket name for inference images.
    minio_client = get_minio_client()
    bucket_name = "inference-data"
    
    # We'll store images in the "inference_images/" prefix within the bucket.
    manifest = []  # To map pid -> MinIO object path
    
    for idx, row in df_ids.iterrows():
        pid = row["pid"]
        vid = row["vid"]  # Still used for coordinate lookup
        print(f"\nProcessing PID: {pid}")
        
        # Step 2: Get coordinates for the record using the vid
        coords = get_coordinates(vid)
        if coords:
            x, y = coords  # Only x and y are needed
            # Step 3: Fetch the aerial image
            image = get_aerial_image(x, y, offset=20)
            if image:
                # Define the object name using the pid (and organized under the "inference_images" prefix)
                object_name = f"inference_images/{pid}.jpg"
                # Immediately upload the image to MinIO
                uploaded_obj = upload_image_to_minio(image, object_name, bucket_name=bucket_name, minio_client=minio_client)
                # Record the mapping in the manifest
                manifest.append({"pid": pid, "minio_object": f"{bucket_name}/{uploaded_obj}"})
            else:
                print(f"❌ Failed to fetch image for PID: {pid}")
        else:
            print(f"❌ Skipping PID: {pid} due to missing coordinates.")
    
    # Step 4: Save the manifest to CSV in data/interim
    manifest_df = pd.DataFrame(manifest)
    manifest_csv = f"data/interim/image_manifest_{gemeentecode}.csv"
    manifest_df.to_csv(manifest_csv, index=False)
    print(f"\nImage manifest saved to {manifest_csv}")

if __name__ == "__main__":
    # Run the pipeline (set demo=True to process only one record for quick testing)
    run_pipeline(gemeentecode="GM0153", limit=50, demo=False)
