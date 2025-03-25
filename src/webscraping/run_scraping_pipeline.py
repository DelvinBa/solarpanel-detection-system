import os
import pandas as pd
from scrape_house_ids import get_pids_and_vids
from scrape_location import get_coordinates
from scrape_images import get_aerial_image

def run_pipeline(gemeentecode="GM0153", limit=10, demo=False):
    """
    Runs the scraping pipeline:
      1. Fetch all house IDs (pid and vid) for a given gemeentecode.
      2. For each record, fetch coordinates and aerial image.
      3. Mimic storing the image in MinIO by saving it locally.
      4. Create a manifest CSV mapping pid and image path.
    
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
    
    # Prepare folder to mimic MinIO storage (local folder)
    image_folder = f"../data/external/images/{gemeentecode}"
    os.makedirs(image_folder, exist_ok=True)
    
    manifest = []  # List to store mapping of pid and image path
    
    for idx, row in df_ids.iterrows():
        pid = row["pid"]
        # We still need the vid to get coordinates, but it won't be used for naming/mapping
        vid = row["vid"]
        print(f"\nProcessing PID: {pid}")
        
        # Step 2: Get coordinates for the record using the vid
        coords = get_coordinates(vid)
        if coords:
            x, y = coords  # Use only the first two values
            # Step 3: Fetch the aerial image, naming the file using only pid
            image_filename = f"{pid}.jpg"
            save_path = os.path.join(image_folder, image_filename)
            image = get_aerial_image(x, y, offset=20, save_path=save_path)
            if image:
                manifest.append({"pid": pid, "image_path": save_path})
        else:
            print(f"Skipping PID: {pid} due to missing coordinates.")
    
    # Step 4: Save the manifest to CSV in data/interim
    manifest_df = pd.DataFrame(manifest)
    manifest_csv = f"../data/interim/image_manifest_{gemeentecode}.csv"
    manifest_df.to_csv(manifest_csv, index=False)
    print(f"\nImage manifest saved to {manifest_csv}")

if __name__ == "__main__":
    # Run the full pipeline for a given city.
    # Use demo=True for quick testing (single record), or set limit to process more.
    run_pipeline(gemeentecode="GM0153", limit=10, demo=True)
