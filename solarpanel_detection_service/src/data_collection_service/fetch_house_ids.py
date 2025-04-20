import pandas as pd
import os

def clean_vid(vid):
    """
    Cleans the vid by:
    - Removing '.0' if present
    - Ensuring it's a string starting with '0'
    """
    if pd.isna(vid):
        return None

    vid_str = str(vid).replace(".0", "").strip()

    if not vid_str.startswith("0"):
        vid_str = "0" + vid_str

    return vid_str

def get_pids_and_vids(gemeentecode):
    # 1) ensure the folder is there
    output_dir = "data/interim"
    os.makedirs(output_dir, exist_ok=True)

    # 2) fetch + clean
    url = f"https://ds.vboenergie.commondatafactory.nl/list/?match-gemeentecode={gemeentecode}"
    df = pd.read_json(url)[['pid', 'vid']].copy()
    df['vid'] = df['vid'].apply(clean_vid).astype(str)
    df['pid'] = df['pid'].astype(str)

    # 3) write to disk
    output_path = os.path.join(output_dir, f"pid_vid_{gemeentecode}.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    return df

# For testing when run standalone:
if __name__ == "__main__":
    gemeentecode = "GM0153"
    df_result = get_pids_and_vids(gemeentecode)
    print(df_result.head(10))