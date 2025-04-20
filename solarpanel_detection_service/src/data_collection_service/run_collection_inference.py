import pandas as pd
import io
from typing import List
from .fetch_house_ids import get_pids_and_vids
from .fetch_location import get_coordinates
from .fetch_images import get_aerial_image
from solarpanel_detection_service.src.minio.minio_utils import upload_image_to_minio, get_minio_client
from solarpanel_detection_service.src.minio.minio_init import init_minio_buckets

def load_manifest(minio_client, bucket_name, manifest_filename="house_id_results.csv"):
    """
    Attempts to load the manifest CSV from MinIO.
    If not found, returns an empty DataFrame with columns: pid, vid, minio_object.
    """
    try:
        response = minio_client.get_object(bucket_name, manifest_filename)
        data = response.read()
        manifest_df = pd.read_csv(io.BytesIO(data), dtype=str)
        print(f"Loaded existing manifest with {len(manifest_df)} entries.")
        return manifest_df
    except Exception as e:
        print("Manifest not found in MinIO. A new manifest will be created.")
        return pd.DataFrame(columns=["pid", "vid", "minio_object"])

def save_manifest(manifest_df, minio_client, bucket_name, manifest_filename="house_id_results.csv"):
    """
    Saves the manifest DataFrame as a CSV and uploads it to MinIO.
    """
    csv_data = manifest_df.to_csv(index=False)
    csv_bytes = csv_data.encode('utf-8')
    minio_client.put_object(bucket_name, manifest_filename, io.BytesIO(csv_bytes), len(csv_bytes))
    print("Manifest uploaded to MinIO as '{}' in bucket '{}'.".format(manifest_filename, bucket_name))

def run_collection_by_city(gemeentecode="GM0153", limit=10):
    """
    Processes houses for a given city:
      1. Fetch house IDs (pid and vid) for the city.
      2. For each record (if not already processed), fetch coordinates and the aerial image.
      3. Upload the image to MinIO (under the inference_images folder).
      4. Append the record (pid, vid, MinIO object path) to the manifest.
      5. Save the updated manifest CSV to MinIO.
    """
    init_minio_buckets()
    minio_client = get_minio_client()
    bucket_name = "inference-data"
    manifest_filename = "house_id_results.csv"
    
    manifest_df = load_manifest(minio_client, bucket_name, manifest_filename)
    df_ids = get_pids_and_vids(gemeentecode).head(limit)
    new_manifest_entries = []
    
    for idx, row in df_ids.iterrows():
        pid = row["pid"]
        vid = row["vid"]
        print(f"Processing PID: {pid}, VID: {vid}")
        
        # Check for duplicates
        if not manifest_df.empty and vid in manifest_df['vid'].values:
            print(f"VID {vid} already exists in manifest.")
            continue
        
        print(f"\nProcessing PID: {pid}")
        coords = get_coordinates(vid)
        if coords:
            x, y = coords
            image = get_aerial_image(x, y, offset=20)
            if image:
                object_name = f"inference_images/{pid}.jpg"
                uploaded_obj = upload_image_to_minio(image, object_name, bucket_name=bucket_name, minio_client=minio_client)
                new_manifest_entries.append({
                    "pid": pid,
                    "vid": vid,
                    "minio_object": f"{bucket_name}/{uploaded_obj}"
                })
            else:
                print(f"❌ Failed to fetch image for PID: {pid}")
        else:
            print(f"❌ Skipping PID: {pid} due to missing coordinates.")
    
    if new_manifest_entries:
        new_entries_df = pd.DataFrame(new_manifest_entries)
        updated_manifest_df = pd.concat([manifest_df, new_entries_df], ignore_index=True)
    else:
        updated_manifest_df = manifest_df

    save_manifest(updated_manifest_df, minio_client, bucket_name, manifest_filename)
    print(f"\nImage manifest updated and stored as '{manifest_filename}' in bucket '{bucket_name}'.")

def run_collection_by_vids(vids: List[str]):
    """
    Processes a list of records based on their VIDs:
      1. For each VID (if not already processed), fetch coordinates and the aerial image.
      2. Upload the image to MinIO and update the manifest.
    """
    init_minio_buckets()
    minio_client = get_minio_client()
    bucket_name = "inference-data"
    manifest_filename = "house_id_results.csv"
    
    manifest_df = load_manifest(minio_client, bucket_name, manifest_filename)
    new_manifest_entries = []
    
    for vid in vids:
        if vid in manifest_df['vid'].values:
            print(f"VID {vid} already exists in manifest. Skipping processing.")
            continue
        
        print(f"\nProcessing VID: {vid}")
        coords = get_coordinates(vid)
        if coords:
            x, y = coords
            image = get_aerial_image(x, y, offset=20)
            if image:
                object_name = f"inference_images/{vid}.jpg"
                uploaded_obj = upload_image_to_minio(image, object_name, bucket_name=bucket_name, minio_client=minio_client)
                new_manifest_entries.append({
                    "pid": None,
                    "vid": vid,
                    "minio_object": f"{bucket_name}/{uploaded_obj}"
                })
            else:
                print(f"❌ Failed to fetch image for VID: {vid}")
        else:
            print(f"❌ Skipping VID: {vid} due to missing coordinates.")
    
    if new_manifest_entries:
        new_entries_df = pd.DataFrame(new_manifest_entries)
        updated_manifest_df = pd.concat([manifest_df, new_entries_df], ignore_index=True)
    else:
        updated_manifest_df = manifest_df
    
    save_manifest(updated_manifest_df, minio_client, bucket_name, manifest_filename)
    print(f"\nImage manifest updated and stored as '{manifest_filename}' in bucket '{bucket_name}'.")

if __name__ == "__main__":
    # Example usage:
    run_collection_by_city(gemeentecode="GM0153", limit=40)
    # run_collection_by_vids(["vid1", "vid2", "vid3"])
